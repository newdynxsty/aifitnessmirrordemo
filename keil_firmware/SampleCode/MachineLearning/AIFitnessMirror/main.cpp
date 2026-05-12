/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Pose landmark network sample. Demonstrate pose landmark detect.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "YOLOv8nPoseModel.hpp"       /* Model API */
#include "YOLOv8nPosePostProcessing.hpp"

// --- NEW REP COUNTER ---
#include "RepCounterModel.hpp" // MUST duplicate ExerciseClassifierModel.hpp/cpp and rename to RepCounterModel
#include "ErrorClassModel.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"
#include "ModelFileReader.h"
#include "ff.h"
#include <math.h>

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_DISPLAY__
#define __USE_UVC__

#include "Profiler.hpp"
#include "ImageSensor.h"

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#define NUM_FRAMEBUF 2  //1 or 2

#define YOLO_MODEL_AT_HYPERRAM_ADDR     (0x82400000)
#define YOLO_MODEL_FILE                "0:\\YOLOv8n-pose.tflite"

// --- NEW REP COUNTER ---
#define REP_COUNTER_MODEL_FILE         "0:\\rep_counter_int8_vela.tflite"
#define ERROR_CLASS_MODEL_FILE         "0:\\workout_error_classifier_int8_vela.tflite"

#define POSE_PRESENCE_THRESHOLD  				(0.7)
#define POSE_KEYPOINT_VISIBLE_THRESHOLD  		(0.5)

// Text Layout Configuration (Right Sidebar)
#define TEXT_X_OFFSET  (320 * IMAGE_DISP_UPSCALE_FACTOR + 10) 
#define TEXT_Y_START   20  // The vertical starting position for the first line
#define LINE_HEIGHT    (FONT_HTIGHT * FONT_DISP_UPSCALE_FACTOR) // Height of one line of text

// Calculate subsequent Y positions automatically
#define TEXT_Y_PRED    (TEXT_Y_START)
#define TEXT_Y_REPS    (TEXT_Y_START + LINE_HEIGHT)
#define TEXT_Y_FPS     (TEXT_Y_START + (2 * LINE_HEIGHT))
#define TEXT_Y_AX    (TEXT_Y_START + (3 * LINE_HEIGHT))
#define TEXT_Y_AY    (TEXT_Y_START + (4 * LINE_HEIGHT))
#define TEXT_Y_AZ    (TEXT_Y_START + (5 * LINE_HEIGHT))
#define TEXT_Y_HR    (TEXT_Y_START + (6 * LINE_HEIGHT))

int timepassed = 0;

// UART1 Bluetooth Buffers
#define RX_BUF_SIZE 128
extern "C" {
    volatile uint8_t g_u8RecData[RX_BUF_SIZE];
    volatile uint32_t g_u32DataIdx = 0;
    volatile bool g_bMsgReceived = false;

    // The IRQ Handler must be extern "C"
    void UART1_IRQHandler(void) {
        uint32_t u32IntSts = UART1->INTSTS;
        if (u32IntSts & UART_INTSTS_RDAINT_Msk) {
            while (!UART_GET_RX_EMPTY(UART1)) {
                uint8_t u8Char = UART_READ(UART1);
                if (g_u32DataIdx < (RX_BUF_SIZE - 1)) {
                    g_u8RecData[g_u32DataIdx++] = u8Char;
                    if (u8Char == '\n' || u8Char == '\r') {
                        g_u8RecData[g_u32DataIdx] = '\0';
                        g_bMsgReceived = true;
                    }
                } else {
                    g_u32DataIdx = 0; // Overflow protection
                }
            }
        }
        if (u32IntSts & (UART_INTSTS_RLSINT_Msk | UART_INTSTS_BUFERRINT_Msk)) {
            UART_ClearIntFlag(UART1, (UART_INTSTS_RLSINT_Msk | UART_INTSTS_BUFERRINT_Msk));
        }
    }
}

static constexpr int KP_WIN = 16;
static constexpr int KP_DIM = 51;
static constexpr int ERROR_CLASS_COUNT = 11;

static const char* const ERROR_CLASS_NAMES[ERROR_CLASS_COUNT] = {
    "JJ ARM LOW",
    "JJ GOOD",
    "JJ LEG NAR",
    "LUNGE GOOD",
    "LUNGE LOW",
    "PUSH GOOD",
    "PUSH KNEE",
    "SIT CORE",
    "SIT GOOD",
    "SQUAT GOOD",
    "SQUAT LOW"
};



// --- NEW REP COUNTER STATE MACHINE GLOBALS ---
static int g_squatCount = 0;
static int g_jumpCount = 0;
static int g_lungeCount = 0;
static int g_pushupCount = 0;
static int g_situpCount = 0;

// --- DEMO STATE GLOBALS ---
static int g_currentRepCount = 0;
static int g_activeExerciseType = 0; // 0=None, 2=Jumping Jack, 5=Situp
static const char* g_activeExerciseName = "WAITING...";

// State Enum: 0 = None, 1 = Squatting, 2 = Jumping, 3 = Lunging, 4 = Pushup, 5 = Situp
static int g_activeExercise = 0;

static inline int8_t QuantizeToInt8(float x, float scale, int32_t zeroPoint)
{
    int32_t v = (int32_t)lrintf(x / scale) + zeroPoint;
    if (v < -128) v = -128;
    if (v > 127)  v = 127;
    return (int8_t)v;
}



enum{
	ePOSE_KP_INDEX_NOSE,				//0
	ePOSE_KP_INDEX_LEFT_EYE,			//1
	ePOSE_KP_INDEX_RIGHT_EYE,			//2
	ePOSE_KP_INDEX_LEFT_EAR,			//3
	ePOSE_KP_INDEX_RIGHT_EAR,			//4
	ePOSE_KP_INDEX_LEFT_SHOULDER,		//5
	ePOSE_KP_INDEX_RIGHT_SHOULDER,		//6
	ePOSE_KP_INDEX_LEFT_ELBOW,			//7
	ePOSE_KP_INDEX_RIGHT_ELBOW,			//8
	ePOSE_KP_INDEX_LEFT_WRIST,			//9
	ePOSE_KP_INDEX_RIGHT_WRIST,			//10
	ePOSE_KP_INDEX_LEFT_HIP,			//11
	ePOSE_KP_INDEX_RIGHT_HIP,			//12
	ePOSE_KP_INDEX_LEFT_KNEE,			//13
	ePOSE_KP_INDEX_RIGHT_KNEE,			//14
	ePOSE_KP_INDEX_LEFT_ANKLE,			//15
	ePOSE_KP_INDEX_RIGHT_ANKLE,			//16
	ePOSE_KP_NUMS,						//17
}E_POSE_KP_INDEX;


typedef enum
{
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct
{
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
    std::vector<arm::app::yolov8n_pose::PoseResult> results;
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;


// --- NEW REP COUNTER --- Memory Arena Allocation
static uint8_t repCounterTensorArena[256 * 1024] ACTIVATION_BUF_ATTRIBUTE; 

// --- ERROR CLASS MODEL --- Memory Arena Allocation
static uint8_t errorClassTensorArena[64 * 1024] ACTIVATION_BUF_ATTRIBUTE;
	
} /* namespace app */
} /* namespace arm */

//frame buffer managemnet function
static S_FRAMEBUF *get_empty_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY)
            return &s_asFramebuf[i];
    }
    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL)
            return &s_asFramebuf[i];
    }
    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;
    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF)
            return &s_asFramebuf[i];
    }
    return NULL;
}

#define IMAGE_DISP_UPSCALE_FACTOR 2
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

#if defined(__USE_UVC__)
#define GLCD_WIDTH	320
#define GLCD_HEIGHT	240
#else
#define GLCD_WIDTH	320 
#define GLCD_HEIGHT	240 
#endif

void Display_PutText_Wrapped(const char* text, int x, int y, uint16_t color, uint16_t bg, int scale) {
    int screenWidth = 800; 
    int charWidth = 8 * scale;   // Standard font is 8 pixels wide
    int charHeight = 16 * scale; // Standard font is 16 pixels tall
    
    int currentX = x;
    int currentY = y;
    
    for (int i = 0; i < strlen(text); i++) {
        if (currentX + charWidth > screenWidth) {
            currentX = x;             
            currentY += charHeight;   
        }
        Display_PutText(&text[i], 1, currentX, currentY, color, bg, false, scale);
        currentX += charWidth;
    }
}
#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".bss.vram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];
#endif

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0 ; i < NUM_FRAMEBUF; i++)
    {
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;
    }

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

static void DrawPoseLandmark(
    const std::vector<arm::app::yolov8n_pose::PoseResult> &results,
    image_t *drawImg
)
{
	arm::app::yolov8n_pose::PoseResult pose;
	int lineColor = COLOR_R5_G6_B5_TO_RGB565(0,COLOR_G6_MAX, 0);	
	int poseSize = results.size();
	
	for(int p = 0; p < poseSize; p ++)
	{
		pose = results[p];
		imlib_draw_rectangle(drawImg, pose.m_poseBox.x, pose.m_poseBox.y, pose.m_poseBox.w, pose.m_poseBox.h, COLOR_B5_MAX, 2, false);

		struct S_KEY_POINT keyPoint;
		struct S_KEY_POINT keyPointTemp;
		std::vector<struct S_KEY_POINT> keyPoints = pose.m_keyPoints;

		for(int k = 0; k < keyPoints.size(); k ++)
		{
			keyPoint = keyPoints[k];
			if(keyPoint.visible < POSE_KEYPOINT_VISIBLE_THRESHOLD)
				continue;

			imlib_draw_circle(drawImg, keyPoint.x, keyPoint.y, 1, COLOR_B5_MAX, 1, true);

			if( k == ePOSE_KP_INDEX_NOSE || k == ePOSE_KP_INDEX_LEFT_SHOULDER) {}
			else if(k == ePOSE_KP_INDEX_LEFT_EYE || k == ePOSE_KP_INDEX_RIGHT_EYE) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_NOSE];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_EAR) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_EYE];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_EAR) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_EYE];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_SHOULDER) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_SHOULDER];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_ELBOW) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_SHOULDER];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_ELBOW) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_SHOULDER];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_WRIST) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_ELBOW];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_RIGHT_WRIST) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_ELBOW];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_HIP) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_SHOULDER];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}			
			else if(k == ePOSE_KP_INDEX_RIGHT_HIP) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_SHOULDER];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_HIP];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}
			else if(k == ePOSE_KP_INDEX_LEFT_KNEE) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_HIP];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}						
			else if(k == ePOSE_KP_INDEX_RIGHT_KNEE) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_HIP];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}			
			else if(k == ePOSE_KP_INDEX_LEFT_ANKLE) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_LEFT_KNEE];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}						
			else if(k == ePOSE_KP_INDEX_RIGHT_ANKLE) {
				keyPointTemp = keyPoints[ePOSE_KP_INDEX_RIGHT_KNEE];
				if(keyPointTemp.visible >=  POSE_KEYPOINT_VISIBLE_THRESHOLD)
					imlib_draw_line(drawImg, keyPoint.x, keyPoint.y, keyPointTemp.x, keyPointTemp.y, lineColor, 1);
			}						
		}
	}
}

static int32_t PrepareModelToHyperRAM(const char* modelFile, uint32_t dstAddr)
{
#define EACH_READ_SIZE 512

    TCHAR sd_path[] = { '0', ':', 0 };
    f_chdrive(sd_path);

    int32_t i32FileSize;
    int32_t i32FileReadIndex = 0;
    int32_t i32Read;

    if (!ModelFileReader_Initialize(modelFile))
    {
        printf_err("Unable open model %s\n", modelFile);
        return -1;
    }

    i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

    while (i32FileReadIndex < i32FileSize)
    {
        i32Read = ModelFileReader_ReadData((BYTE *)(dstAddr + i32FileReadIndex), EACH_READ_SIZE);
        if (i32Read < 0)
            break;
        i32FileReadIndex += i32Read;
    }

    ModelFileReader_Finish();

    if (i32FileReadIndex < i32FileSize)
    {
        printf_err("Read model file size is not enough\n");
        return -2;
    }

    return i32FileSize;
}

int main()
{
	/* Initialise the UART module to allow printf related functions (if using retarget) */
	BoardInit();
	SYS_UnlockReg();
	/* UART1 (Bluetooth) Clock & Pins */
	CLK_SetModuleClock(UART1_MODULE, CLK_UARTSEL0_UART1SEL_HIRC, CLK_UARTDIV0_UART1DIV(1));
	CLK_EnableModuleClock(UART1_MODULE);
	SET_UART1_RXD_PA2();
	SET_UART1_TXD_PA3();

	/* UART1 Hardware Init */
	SYS_ResetModule(SYS_UART1RST);
	UART_Open(UART1, 9600);
	UART_EnableInt(UART1, UART_INTEN_RDAIEN_Msk);
	NVIC_EnableIRQ(UART1_IRQn);
	SYS_LockReg();
			
	/* Copy model file from SD to HyperRAM*/		
	int32_t yoloModelSize = PrepareModelToHyperRAM(YOLO_MODEL_FILE, YOLO_MODEL_AT_HYPERRAM_ADDR);
	if (yoloModelSize <= 0)
	{
			printf_err("Failed to prepare YOLO model\n");
			return 1;
	}

    // --- NEW REP COUNTER --- Compute Address and Load Model
    uint32_t repCounterModelAddr = YOLO_MODEL_AT_HYPERRAM_ADDR + ((yoloModelSize + 0xFFF) & ~0xFFF);
    int32_t repCounterModelSize = PrepareModelToHyperRAM(REP_COUNTER_MODEL_FILE, repCounterModelAddr);
    if (repCounterModelSize <= 0)
    {
            printf_err("Failed to prepare REP COUNTER model\n");
            return 1;
    }

    uint32_t errorClassModelAddr = repCounterModelAddr + ((repCounterModelSize + 0xFFF) & ~0xFFF);
    int32_t errorClassModelSize = PrepareModelToHyperRAM(ERROR_CLASS_MODEL_FILE, errorClassModelAddr);
    if (errorClassModelSize <= 0)
    {
            printf_err("Failed to prepare ERROR CLASS model\n");
            return 1;
    }

	// --- Init YOLO model interpreter ---
	arm::app::YOLOv8nPoseModel poseModel;
	if (!poseModel.Init(arm::app::tensorArena,
						sizeof(arm::app::tensorArena),
						(unsigned char*)YOLO_MODEL_AT_HYPERRAM_ADDR,
						yoloModelSize))
	{
			printf_err("Failed to initialise YOLO model\n");
			return 1;
	}


    // --- NEW REP COUNTER --- Init Model Interpreter
    arm::app::RepCounterModel repCounterModel;
    if (!repCounterModel.Init(arm::app::repCounterTensorArena,
                              sizeof(arm::app::repCounterTensorArena),
                              (unsigned char*)repCounterModelAddr,
                              repCounterModelSize))
    {
            printf_err("Failed to initialise REP COUNTER model\n");
            return 1;
    }

    arm::app::ErrorClassModel errorClassModel;
    if (!errorClassModel.Init(arm::app::errorClassTensorArena,
                              sizeof(arm::app::errorClassTensorArena),
                              (unsigned char*)errorClassModelAddr,
                              errorClassModelSize))
    {
            printf_err("Failed to initialise ERROR CLASS model\n");
            return 1;
    }

    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena), ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena) + ACTIVATION_BUF_SZ - 1), eMPU_ATTR_CACHEABLE_WTRA)
        },

        // --- NEW REP COUNTER --- MPU Configuration
        {
            ARM_MPU_RBAR(((unsigned int)arm::app::repCounterTensorArena), ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((((unsigned int)arm::app::repCounterTensorArena) + sizeof(arm::app::repCounterTensorArena) - 1), eMPU_ATTR_CACHEABLE_WTRA)
        },
        {
            ARM_MPU_RBAR(((unsigned int)arm::app::errorClassTensorArena), ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((((unsigned int)arm::app::errorClassTensorArena) + sizeof(arm::app::errorClassTensorArena) - 1), eMPU_ATTR_CACHEABLE_WTRA)
        },
        {
            ARM_MPU_RBAR(((unsigned int)fb_array), ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1), eMPU_ATTR_NON_CACHEABLE)
        },
#if (NUM_FRAMEBUF == 2)
        {
            ARM_MPU_RBAR(((unsigned int)frame_buf1), ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + OMV_FB_SIZE - 1), eMPU_ATTR_NON_CACHEABLE)
        },
#endif
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    TfLiteTensor *inputTensor   = poseModel.GetInputTensor(0);

    TfLiteIntArray *inputShape = poseModel.GetInputShape(0);
    const int inputImgCols = inputShape->data[arm::app::YOLOv8nPoseModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::YOLOv8nPoseModel::ms_inputRowsIdx];
    const uint32_t nChannels = inputShape->data[arm::app::YOLOv8nPoseModel::ms_inputChannelsIdx];

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);
		


    // --- NEW REP COUNTER --- Tensor Extractions
    TfLiteTensor* repInTensor = repCounterModel.GetInputTensor(0);
    TfLiteTensor* repOutTensor = repCounterModel.GetOutputTensor(0);
    arm::app::QuantParams repInQ = arm::app::GetTensorQuantParams(repInTensor);
    arm::app::QuantParams repOutQ = arm::app::GetTensorQuantParams(repOutTensor);

    TfLiteTensor* errorInTensor = errorClassModel.GetInputTensor(0);
    TfLiteTensor* errorOutTensor = errorClassModel.GetOutputTensor(0);
    arm::app::QuantParams errorInQ = arm::app::GetTensorQuantParams(errorInTensor);
    arm::app::QuantParams errorOutQ = arm::app::GetTensorQuantParams(errorOutTensor);

    // postProcess
	arm::app::yolov8n_pose::YOLOv8nPosePostProcessing postProcess(&poseModel, POSE_PRESENCE_THRESHOLD);
	
    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

#ifndef __PROFILE__
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 1
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);

#if defined (__USE_DISPLAY__)
    char szDisplayText[100];
    S_DISP_RECT sDispRect;
    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
	UVC_Init();
    HSUSBD_Start();
#endif

    while (1)
    {
        emptyFramebuf = get_empty_framebuf();
        if (emptyFramebuf)
        {
            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
		}
		
        fullFramebuf = get_full_framebuf();
        if (fullFramebuf)
        {
            image_t resizeImg;
            roi.x = 0; roi.y = 0; roi.w = fullFramebuf->frameImage.w; roi.h = fullFramebuf->frameImage.h;

            resizeImg.w = inputImgCols; resizeImg.h = inputImgRows;
            resizeImg.data = (uint8_t *)inputTensor->data.data;
            resizeImg.pixfmt = PIXFORMAT_RGB888;

            imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg, &roi);

			auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
			auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

			for (size_t i = 0; i < inputTensor->bytes; i++)
			{
				signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
			}

			poseModel.RunInference();
            fullFramebuf->eState = eFRAMEBUF_INF;
        }

        infFramebuf = get_inf_framebuf();
        if (infFramebuf)
        {
			postProcess.RunPostProcessing(
				inputImgCols, inputImgRows, infFramebuf->frameImage.w, infFramebuf->frameImage.h, infFramebuf->results);
					
			// ---- EXERCISE CLASSIFIER AND REP COUNTER ----

					  float current_pose_conf = 0.0f;
						float prob_jump_middle = 0.0f;
						float prob_lunge_middle = 0.0f;
						float prob_pushup_middle = 0.0f;
						float prob_pushup_start = 0.0f;
						float prob_situp_middle = 0.0f;
						float prob_situp_start = 0.0f;
						float prob_squat_middle = 0.0f;
						float prob_squat_start = 0.0f;
						int current_pose_class = 0;
						float current_error_conf = 0.0f;
						int current_error_class = -1;

            if (infFramebuf->results.size() > 0)
            {
                const auto& pose = infFramebuf->results[0];

                float feat[KP_DIM];
                int idx = 0;

                for (int k = 0; k < ePOSE_KP_NUMS; k++)
                {
                    const auto& kp = pose.m_keyPoints[k];
                    
                    // Normalize relative to the bounding box rather than the full screen
                    float box_w = (pose.m_poseBox.w > 0) ? (float)pose.m_poseBox.w : 1.0f;
                    float box_h = (pose.m_poseBox.h > 0) ? (float)pose.m_poseBox.h : 1.0f;

                    float xn = (float)(kp.x - pose.m_poseBox.x) / box_w;
                    float yn = (float)(kp.y - pose.m_poseBox.y) / box_h;
                    float cn = kp.visible;

                    feat[idx++] = xn;
                    feat[idx++] = yn;
                    feat[idx++] = cn;
                }



                // --- NEW REP COUNTER --- 
                int8_t* repIn = (int8_t*)repInTensor->data.data;
                for (int i = 0; i < KP_DIM; i++) {
                    repIn[i] = QuantizeToInt8(feat[i], repInQ.scale, repInQ.offset);
                }

                repCounterModel.RunInference();

                int8_t* errorIn = (int8_t*)errorInTensor->data.data;
                for (int i = 0; i < KP_DIM; i++) {
                    errorIn[i] = QuantizeToInt8(feat[i], errorInQ.scale, errorInQ.offset);
                }

                errorClassModel.RunInference();

                int8_t* errorOut = (int8_t*)errorOutTensor->data.data;
                float error_out_scale = (errorOutQ.scale <= 0.0f) ? (1.0f / 255.0f) : errorOutQ.scale;
                int error_out_offset = (errorOutQ.scale <= 0.0f) ? -128 : errorOutQ.offset;
                // We will compute current_error_class after updating g_activeExerciseType

                // Extract INT8 outputs and Dequantize
                int8_t* repOut = (int8_t*)repOutTensor->data.data;
                
                // Fallback for Softmax INT8 quantization if Vela stripped the quant params
                float out_scale = (repOutQ.scale <= 0.0f) ? (1.0f / 255.0f) : repOutQ.scale;
                int out_offset = (repOutQ.scale <= 0.0f) ? -128 : repOutQ.offset;

								prob_jump_middle   = ((int)repOut[0] - out_offset) * out_scale;
								prob_lunge_middle  = ((int)repOut[1] - out_offset) * out_scale;
								prob_pushup_middle = ((int)repOut[2] - out_offset) * out_scale;
								prob_pushup_start  = ((int)repOut[3] - out_offset) * out_scale;
								prob_situp_middle  = ((int)repOut[4] - out_offset) * out_scale;
								prob_situp_start   = ((int)repOut[5] - out_offset) * out_scale;
								prob_squat_middle  = ((int)repOut[6] - out_offset) * out_scale;
								prob_squat_start   = ((int)repOut[7] - out_offset) * out_scale;
                // Find highest probability
                int current_pose_class = 0;
                current_pose_conf = prob_jump_middle;
                
                if (prob_lunge_middle > current_pose_conf) { current_pose_class = 1; current_pose_conf = prob_lunge_middle; }
                if (prob_pushup_middle > current_pose_conf) { current_pose_class = 2; current_pose_conf = prob_pushup_middle; }
                if (prob_pushup_start > current_pose_conf) { current_pose_class = 3; current_pose_conf = prob_pushup_start; }
                if (prob_situp_middle > current_pose_conf) { current_pose_class = 4; current_pose_conf = prob_situp_middle; }
                if (prob_situp_start > current_pose_conf) { current_pose_class = 5; current_pose_conf = prob_situp_start; }
                if (prob_squat_middle > current_pose_conf) { current_pose_class = 6; current_pose_conf = prob_squat_middle; }
                if (prob_squat_start > current_pose_conf) { current_pose_class = 7; current_pose_conf = prob_squat_start; }

								// --- DEMO COUNTER: JUMPING JACKS, SITUPS & SQUATS ---
								if (current_pose_conf >= 0.70f) {
										
										// 1. IDENTIFY THE EXERCISE & HANDLE AUTO-RESET
										int detectedType = 0; 
										if (current_pose_class == 0) detectedType = 2;      // Jumping Jack (middle)
										else if (current_pose_class == 4 || current_pose_class == 5) detectedType = 5; // Situp (middle)
										else if (current_pose_class == 6) detectedType = 1; // Squat (middle)
										else if (current_pose_class == 1) detectedType = 3; // Lunge (middle)
										else if (current_pose_class == 2 || current_pose_class == 3) detectedType = 4; // Pushup (middle)

										// If we hit a 'middle' state of a different exercise, reset the counter
										if (detectedType != 0 && g_activeExerciseType != detectedType) {
												g_activeExerciseType = detectedType;
												g_currentRepCount = 0;
												if (detectedType == 2) g_activeExerciseName = "JUMPING JACK";
												else if (detectedType == 5) g_activeExerciseName = "SIT-UP";
												else if (detectedType == 1) g_activeExerciseName = "SQUAT";
												else if (detectedType == 3) g_activeExerciseName = "LUNGE";
												else if (detectedType == 4) g_activeExerciseName = "PUSH-UP";
												g_activeExercise = 0; // Reset state machine phase
											
												// Clear the text area to prevent character overlapping
												S_DISP_RECT sClearRect;
												sClearRect.u32TopLeftX = 650; 
												sClearRect.u32TopLeftY = 100;
												sClearRect.u32BottonRightX = 800;
												sClearRect.u32BottonRightY = 160;
												Display_ClearRect(C_WHITE, &sClearRect);
										}

										// 2. REP COUNTING
										// STAND
										if (current_pose_class == 7) {
												if (g_activeExercise == 2 || g_activeExercise == 1 || g_activeExercise == 3) g_currentRepCount++;
												g_activeExercise = 0;
										} 
										// JUMP_MIDDLE
										else if (current_pose_class == 0) {
												g_activeExercise = 2; 
										}
										// SQUAT_MIDDLE
										else if (current_pose_class == 6) {
												g_activeExercise = 1; 
										}
										// SITUP_START
										else if (current_pose_class == 5) {
												if (g_activeExercise == 5) g_currentRepCount++;
												g_activeExercise = 0;
										}
										// SITUP_MIDDLE
										else if (current_pose_class == 4) {
												g_activeExercise = 5;
										}
										// LUNGE MIDDLE
										else if (current_pose_class == 1) {
												g_activeExercise = 3;
										}
										// PUSHUP_START
										else if (current_pose_class == 3) {
											if (g_activeExercise == 4) g_currentRepCount++;
											g_activeExercise = 0;
										}
										// PUSHUP_MIDDLE
										else if (current_pose_class == 2) {
											g_activeExercise = 4;
										}
								}
								
								// --- EVALUATE ERROR CLASS BASED ON CURRENT EXERCISE ---
								current_error_class = -1;
								current_error_conf = -1.0f;
								
								for (int i = 0; i < ERROR_CLASS_COUNT; i++) {
										float p = ((int)errorOut[i] - error_out_offset) * error_out_scale;
										
										// Filter by active exercise
										bool match = false;
										if (g_activeExerciseType == 2 && strncmp(ERROR_CLASS_NAMES[i], "JJ", 2) == 0) match = true;
										else if (g_activeExerciseType == 5 && strncmp(ERROR_CLASS_NAMES[i], "SIT", 3) == 0) match = true;
										else if (g_activeExerciseType == 1 && strncmp(ERROR_CLASS_NAMES[i], "SQUAT", 5) == 0) match = true;
										else if (g_activeExerciseType == 0) match = true; // No active exercise, consider all
										
										if (match && p > current_error_conf) {
												current_error_class = i;
												current_error_conf = p;
										}
								}
								// Default to 0 if nothing matched
								if (current_error_class == -1) {
										current_error_class = 0;
										current_error_conf = 0.0f;
								}
						}

            //draw bbox and render
						if(infFramebuf->results.size())
						{
							DrawPoseLandmark(infFramebuf->results, &infFramebuf->frameImage);
						}

						//display result image
						#if defined (__USE_DISPLAY__)
									sDispRect.u32TopLeftX = 0;
									sDispRect.u32TopLeftY = 0;
									sDispRect.u32BottonRightX = ((frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
									sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

									Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect, IMAGE_DISP_UPSCALE_FACTOR);
						
						#if defined (__USE_UVC__)
							if (UVC_IsConnect()) // Check if the PC has opened the camera app
							{
									image_t origImg; origImg.w=infFramebuf->frameImage.w; origImg.h=infFramebuf->frameImage.h;
									origImg.data = (uint8_t *)infFramebuf->frameImage.data; origImg.pixfmt = PIXFORMAT_RGB565;
									image_t vflipImg = origImg; imlib_nvt_vflip(&origImg, &vflipImg);
									UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE, uvcStatus.StillImage);
							}
						#endif

						// --- DEMO DISPLAY ---
						char displayBuffer[64];

						// Display Exercise Name
						snprintf(displayBuffer, sizeof(displayBuffer), "%s", g_activeExerciseName);
						Display_PutText_Wrapped(
								displayBuffer,
								650, 100,
								C_RED, C_WHITE,
								FONT_DISP_UPSCALE_FACTOR
						);

						// Display Rep Count (Large)
						snprintf(displayBuffer, sizeof(displayBuffer), "REPS: %d", g_currentRepCount);
						Display_PutText_Wrapped(
								displayBuffer,
								650, 160,
								C_BLUE, C_WHITE,
								FONT_DISP_UPSCALE_FACTOR
						);

						// Debug Squat Probabilities
						char debugBuffer[128];
						snprintf(debugBuffer, sizeof(debugBuffer), "SqM:%.2f SqS:%.2f", prob_squat_middle, prob_squat_start);
						Display_PutText_Wrapped(debugBuffer, 650, 220, C_BLUE, C_WHITE, 1);

						// Display Confidence
						snprintf(displayBuffer, sizeof(displayBuffer), "C: %.2f", current_pose_conf);
						Display_PutText_Wrapped(displayBuffer, 650, 260, C_BLACK, C_WHITE, FONT_DISP_UPSCALE_FACTOR);

						// Clear error text area before redrawing
						S_DISP_RECT sErrClearRect;
						sErrClearRect.u32TopLeftX = 650;
						sErrClearRect.u32TopLeftY = 320;
						sErrClearRect.u32BottonRightX = 800;
						sErrClearRect.u32BottonRightY = 416;
						Display_ClearRect(C_WHITE, &sErrClearRect);

						const char* errorName = "NO POSE";
						if (current_error_class >= 0 && current_error_class < ERROR_CLASS_COUNT) {
								errorName = ERROR_CLASS_NAMES[current_error_class];
						}
						snprintf(displayBuffer, sizeof(displayBuffer), "%s", errorName);
						Display_PutText_Wrapped(displayBuffer, 650, 320, C_MAGENTA, C_WHITE, FONT_DISP_UPSCALE_FACTOR);

						snprintf(displayBuffer, sizeof(displayBuffer), "EC: %.2f", current_error_conf);
						Display_PutText_Wrapped(displayBuffer, 650, 380, C_MAGENTA, C_WHITE, FONT_DISP_UPSCALE_FACTOR);
						
						// Send exercise data over serial
						printf("DATA:%s,%.2f,%d,%s,%.2f\n", 
							g_activeExerciseName,      // Current Exercise Name (String)
							current_pose_conf,         // Exercise Confidence (Float)
							g_currentRepCount,         // Rep Count (Int)
							errorName,                 // Bad Form Class Name (String)
							current_error_conf         // Bad Form Confidence (Float)
						);
			// Display accelerometer and heart rate data
			/**
			if (g_bMsgReceived) {
						char localBuf[RX_BUF_SIZE];
						NVIC_DisableIRQ(UART1_IRQn);
						memcpy(localBuf, (const void*)g_u8RecData, g_u32DataIdx);
						g_u32DataIdx = 0;
						g_bMsgReceived = false;
						NVIC_EnableIRQ(UART1_IRQn);

						float bAx = 0, bAy = 0, bAz = 0;
						char bHr[20] = {0};
						
						if (sscanf(localBuf, "A:%f,%f,%f|HR:%s", &bAx, &bAy, &bAz, bHr) >= 3) {
										S_DISP_RECT sRect;
										sRect.u32TopLeftX = TEXT_X_OFFSET;
										sRect.u32BottonRightX = Disaplay_GetLCDWidth() - 1;
										
										sprintf(szDisplayText, "AX: %.2f", bAx);
										sRect.u32TopLeftY = TEXT_Y_AX; sRect.u32BottonRightY = TEXT_Y_AX + LINE_HEIGHT - 1;
										Display_ClearRect(C_WHITE, &sRect);
										Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_AX, C_BLACK, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);

										sprintf(szDisplayText, "AY: %.2f", bAy);
										sRect.u32TopLeftY = TEXT_Y_AY; sRect.u32BottonRightY = TEXT_Y_AY + LINE_HEIGHT - 1;
										Display_ClearRect(C_WHITE, &sRect);
										Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_AY, C_BLACK, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);

										sprintf(szDisplayText, "AZ: %.2f", bAz);
										sRect.u32TopLeftY = TEXT_Y_AZ; sRect.u32BottonRightY = TEXT_Y_AZ + LINE_HEIGHT - 1;
										Display_ClearRect(C_WHITE, &sRect);
										Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_AZ, C_BLACK, C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);
										
										timepassed +=1;
										sprintf(szDisplayText, "HR: %s", bHr);
										sRect.u32TopLeftY = TEXT_Y_HR; sRect.u32BottonRightY = TEXT_Y_HR + LINE_HEIGHT - 1;
										Display_ClearRect(C_WHITE, &sRect);
										Display_PutText(szDisplayText, strlen(szDisplayText), TEXT_X_OFFSET, TEXT_Y_HR, (strcmp(bHr, "No") == 0 ? C_RED : C_MAGENTA), C_WHITE, false, FONT_DISP_UPSCALE_FACTOR);
						}
				}
				**/
#endif

            u64PerfFrames ++;
			if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
#if defined (__USE_DISPLAY__)
				#define SCREEN_HEIGHT 240			
				uint32_t yOffset = (SCREEN_HEIGHT - (frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR)) / 2;
				sDispRect.u32TopLeftX = 0;																																																																																																													
				sDispRect.u32TopLeftY = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
				sDispRect.u32BottonRightX = (frameBuffer.w);
				sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
								sprintf(szDisplayText, "FPS: %llu", u64PerfFrames / EACH_PERF_SEC);
                Display_PutText_Wrapped(
                        szDisplayText,
                        650,
                        0,
                        C_BLUE,
                        C_WHITE,
                        FONT_DISP_UPSCALE_FACTOR
                );
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
			}
            infFramebuf->eState = eFRAMEBUF_EMPTY;
		}

		if (emptyFramebuf)
		{
			ImageSensor_WaitCaptureDone();
            emptyFramebuf->eState = eFRAMEBUF_FULL;		
		}
    }
    return 0;
}