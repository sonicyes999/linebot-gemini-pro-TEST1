Temperture 和 ChatTemperture 改為 ImageTemperature 和 ChatTemperature。
Context: 對於長時間運行的服務（例如 Web 伺服器），考慮傳遞來自請求的 context 或使用 context.WithTimeout 而不是總是使用 context.Background()。
註解: 可以添加更多中文註解來解釋程式碼的特定部分。
修正後的範例 (僅修改錯誤處理和拼寫)
package main

import (
	"context"
	"errors" // 需要引入 errors
	"fmt"
	"log"
	"os" // 假設從環境變數讀取金鑰

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

const ImageTemperature = 0.8 // 修正拼寫
const ChatTemperature = 0.3  // 修正拼寫

// 從環境變數讀取 API Key，你需要設定 "GEMINI_API_KEY" 這個環境變數
var geminiKey = os.Getenv("GEMINI_API_KEY")

// GeminiImage: 輸入圖像數據，獲取模型的文字描述。
func GeminiImage(imgData []byte) (string, error) {
	if geminiKey == "" {
		return "", errors.New("GEMINI_API_KEY 環境變數未設定")
	}
	ctx := context.Background()
	// 使用 API 金鑰初始化客戶端
	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		// 返回錯誤而不是終止程式
		log.Printf("無法建立 genai 客戶端: %v\n", err)
		return "", fmt.Errorf("無法建立 genai 客戶端: %w", err)
	}
	// 確保函式結束時關閉客戶端
	defer client.Close()

	// 選擇模型
	model := client.GenerativeModel("gemini-1.5-flash")
	// 設定溫度
	value := float32(ImageTemperature)
	model.Temperature = &value
	// 準備請求內容 (圖像 + 文字提示)
	prompt := []genai.Part{
		genai.ImageData("png", imgData), // 假設是 PNG 格式
		genai.Text("用科學的細節描述這張圖片，並以繁體中文(zh-TW)回覆："),
	}
	log.Println("開始處理圖像...")
	// 發送請求給模型
	resp, err := model.GenerateContent(ctx, prompt...)
	if err != nil {
		// 返回錯誤而不是終止程式
		log.Printf("生成內容時出錯: %v\n", err)
		return "", fmt.Errorf("生成內容時出錯: %w", err)
	}
	log.Println("圖像處理完成.") // 移除 resp 打印，printResponse 會處理

	// 提取並返回文字回應
	return printResponse(resp), nil
}

// startNewChatSession: 啟動一個新的聊天會話。
// 返回聊天會話物件和可能的錯誤。
func startNewChatSession() (*genai.ChatSession, error) {
	if geminiKey == "" {
		return nil, errors.New("GEMINI_API_KEY 環境變數未設定")
	}
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		log.Printf("無法建立 genai 客戶端: %v\n", err)
		// 返回錯誤
		return nil, fmt.Errorf("無法建立 genai 客戶端: %w", err)
	}
	// 注意：此處未關閉 client。如果聊天會話生命週期很長，
	// 可能需要考慮客戶端的管理方式，或者讓呼叫者管理 client。
	// 對於簡單應用，模型會處理。

	model := client.GenerativeModel("gemini-1.5-flash")
	value := float32(ChatTemperature)
	model.Temperature = &value
	// 開始一個新的聊天，記住歷史記錄
	cs := model.StartChat()
	log.Println("已啟動新的聊天會話。")
	// 返回會話和 nil 錯誤
	return cs, nil
}

// send: 向指定的聊天會話發送訊息。
// 返回模型的回應和可能的錯誤。
func send(cs *genai.ChatSession, msg string) (*genai.GenerateContentResponse, error) {
	// 如果會話為 nil，嘗試創建一個新的
	if cs == nil {
		var err error
		log.Println("聊天會話為 nil，嘗試啟動新的會話...")
		cs, err = startNewChatSession()
		if err != nil {
			// 如果創建新會話失敗，直接返回錯誤
			return nil, fmt.Errorf("無法啟動新的聊天會話: %w", err)
		}
	}

	ctx := context.Background()
	log.Printf("== 我: %s\n== 模型:\n", msg)
	// 發送訊息到會話
	res, err := cs.SendMessage(ctx, genai.Text(msg))
	if err != nil {
		log.Printf("發送訊息時出錯: %v\n", err)
		// 返回錯誤
		return nil, fmt.Errorf("發送訊息時出錯: %w", err)
	}
	// 返回回應和 nil 錯誤
	return res, nil
}

// printResponse: 從 GenerateContentResponse 中提取文字內容。
func printResponse(resp *genai.GenerateContentResponse) string {
	var ret string
	if resp == nil || resp.Candidates == nil {
		log.Println("收到的回應為 nil 或沒有候選內容。")
		return ""
	}
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				ret = ret + fmt.Sprintf("%v", part)
				// 可以在這裡記錄每個部分，如果需要詳細日誌
				// log.Printf("  - Part: %v\n", part)
			}
		}
	}
	if ret == "" {
		log.Println("從回應中未提取到文字內容。")
	}
	return ret
}

// --- main 函式範例 (需要您根據實際需求實現) ---
// func main() {
// 	// --- 圖像處理範例 ---
// 	// 讀取你的圖像文件 (例如: "image.png")
// 	// imgBytes, err := os.ReadFile("path/to/your/image.png")
// 	// if err != nil {
// 	// 	log.Fatalf("無法讀取圖像文件: %v", err)
// 	// }
// 	// description, err := GeminiImage(imgBytes)
// 	// if err != nil {
// 	// 	log.Printf("圖像處理失敗: %v\n", err)
// 	// } else {
// 	// 	fmt.Println("=== 圖像描述 ===")
// 	// 	fmt.Println(description)
// 	// }
//
// 	// --- 聊天範例 ---
// 	cs, err := startNewChatSession()
// 	if err != nil {
// 		log.Fatalf("無法開始聊天會話: %v", err)
// 	}
//
// 	// 第一輪對話
// 	resp1, err := send(cs, "你好！請用繁體中文介紹一下你自己。")
// 	if err != nil {
// 		log.Printf("第一次發送失敗: %v\n", err)
// 	} else {
// 		fmt.Println(printResponse(resp1))
// 	}
//
// 	// 第二輪對話 (會基於上一輪的內容)
// 	resp2, err := send(cs, "台灣最高的山是哪一座？")
// 	if err != nil {
// 		log.Printf("第二次發送失敗: %v\n", err)
// 	} else {
// 		fmt.Println(printResponse(resp2))
// 	}
// }
