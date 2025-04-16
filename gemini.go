package main

import (
	"context"
	"fmt"
	"log"
	"os" // 用於讀取環境變數

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// 設定模型溫度參數
const ImageTemperature = 0.8 // 圖片處理溫度
const ChatTemperature = 0.3  // 聊天溫度

// --- 全域變數 (或在 main 中初始化後傳遞) ---
// 註解掉，改為在 main 中處理
// var geminiKey string = os.Getenv("GEMINI_API_KEY")

// GeminiImage: 輸入圖片數據，返回模型的文字描述
// 接收 context, client 和圖片數據
func GeminiImage(ctx context.Context, client *genai.Client, imgData []byte, imageType string) (string, error) {
	// Client 在 main 函數中建立並傳入，這裡不再建立

	model := client.GenerativeModel("gemini-1.5-flash") // 或者 gemini-pro-vision
	value := float32(ImageTemperature)
	model.Temperature = &value

	// 檢查圖片類型是否支援，預設為 "png"
	// 常見的有 "png", "jpeg", "webp", "heic", "heif"
	if imageType == "" {
		imageType = "png"
	}

	prompt := []genai.Part{
		genai.ImageData(imageType, imgData), // 使用傳入的圖片類型
		genai.Text("請以科學的細節描述這張圖片，並以繁體中文(zh-TW)回覆:"),
	}

	log.Println("開始處理圖片...")
	resp, err := model.GenerateContent(ctx, prompt...)
	if err != nil {
		log.Printf("處理圖片時產生錯誤: %v", err) // 改為記錄錯誤並返回
		return "", fmt.Errorf("模型生成內容失敗: %w", err)
	}
	log.Println("圖片處理完成.") // 移除顯示 resp 內容，通常很長

	return printResponse(resp), nil // 返回處理結果
}

// startNewChatSession: 使用提供的客戶端啟動一個新的聊天會話
func startNewChatSession(ctx context.Context, client *genai.Client) (*genai.ChatSession, error) {
	// Client 在 main 函數中建立並傳入，這裡不再建立

	model := client.GenerativeModel("gemini-1.5-flash") // 或者 gemini-pro
	value := float32(ChatTemperature)
	model.Temperature = &value
	cs := model.StartChat()
	// StartChat 本身不直接返回錯誤，模型設定等問題會在 client 建立時或 SendMessage 時體現
	log.Println("已啟動新的聊天會話。")
	return cs, nil
}

// send: 向聊天會話發送訊息
// 返回模型的回應文字、更新後的會話 (如果原本是 nil) 和錯誤
func send(ctx context.Context, client *genai.Client, cs *genai.ChatSession, msg string) (string, *genai.ChatSession, error) {
	var err error
	// 如果會話是 nil，使用傳入的 client 啟動一個新的
	if cs == nil {
		log.Println("偵測到空的聊天會話，正在啟動新的會話...")
		cs, err = startNewChatSession(ctx, client)
		if err != nil {
			return "", nil, fmt.Errorf("無法啟動新的聊天會話: %w", err)
		}
	}

	log.Printf("== 我: %s\n", msg)
	res, err := cs.SendMessage(ctx, genai.Text(msg))
	if err != nil {
		log.Printf("發送訊息時產生錯誤: %v", err) // 改為記錄錯誤並返回
		return "", cs, fmt.Errorf("傳送訊息失敗: %w", err)
	}

	log.Println("== 模型:") // 回應內容會在 printResponse 中打印
	response_text := printResponse(res)
	return response_text, cs, nil // 返回回應文字、當前會話和 nil 錯誤
}

// printResponse: 從 GenerateContentResponse 中提取並打印文字回應
func printResponse(resp *genai.GenerateContentResponse) string {
	// 增加健壯性檢查
	if resp == nil || resp.Candidates == nil || len(resp.Candidates) == 0 {
		log.Println("收到的回應為空或無有效候選內容。")
		return ""
	}

	var ret string
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				ret = ret + fmt.Sprintf("%v", part)
				log.Printf("   - %v\n", part) // 在日誌中打印模型回應部分
			}
		}
	}
	// 移除結尾可能多餘的換行符（如果有的話）
	// ret = strings.TrimSpace(ret)
	return ret
}

// main 函數: 程式入口點
func main() {
	// 1. 讀取 API 金鑰
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if geminiKey == "" {
		log.Fatal("錯誤：請設定環境變數 GEMINI_API_KEY") // 如果未設定金鑰則終止
	}

	// 2. 建立 Context
	ctx := context.Background()

	// 3. 建立 Gemini 客戶端 (一次性)
	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		log.Fatalf("建立 Gemini 客戶端失敗: %v", err)
	}
	defer client.Close() // 確保程式結束時關閉客戶端
	log.Println("Gemini 客戶端建立成功。")

	// --- 範例：使用 GeminiImage (需要您提供圖片數據) ---
	log.Println("\n--- 圖片處理範例 ---")
	// // **** 取消註解並替換這裡來載入您的圖片 ****
	// // 例如：從檔案讀取
	// imgBytes, err := os.ReadFile("path/to/your/image.png") // 請替換為您的圖片路徑
	// if err != nil {
	//  log.Printf("讀取圖片檔案失敗: %v", err)
	// } else {
	//  imageDesc, err := GeminiImage(ctx, client, imgBytes, "png") // 指定圖片類型，例如 "png" 或 "jpeg"
	//  if err != nil {
	//      log.Printf("圖片處理失敗: %v", err)
	//  } else {
	//      log.Printf("模型對圖片的描述:\n%s\n", imageDesc)
	//  }
	// }
	log.Println("圖片處理範例區塊結束（目前已註解，請自行載入圖片數據）。")

	// --- 範例：使用聊天功能 ---
	log.Println("\n--- 聊天功能範例 ---")
	var chatSession *genai.ChatSession // 初始化為 nil

	// 發送第一條訊息 (會自動啟動新會話)
	msg1 := "你好，Gemini！請用繁體中文回答我。"
	resp1, updatedSession, err := send(ctx, client, chatSession, msg1)
	if err != nil {
		log.Printf("第一次發送訊息失敗: %v", err)
	} else {
		chatSession = updatedSession // 更新會話狀態
		log.Printf("模型第一次回應: %s", resp1)
	}

	// 如果第一次成功，發送第二條訊息 (使用現有會話)
	if chatSession != nil {
		msg2 := "請告訴我一個關於 Go 語言的笑話。"
		resp2, updatedSession2, err := send(ctx, client, chatSession, msg2)
		if err != nil {
			log.Printf("第二次發送訊息失敗: %v", err)
		} else {
			chatSession = updatedSession2 // 再次更新會話狀態 (雖然此例中是同一個)
			log.Printf("模型第二次回應: %s", resp2)
		}
	}

	log.Println("\n--- 範例結束 ---")
}
