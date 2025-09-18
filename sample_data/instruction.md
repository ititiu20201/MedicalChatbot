Step-by-step Instructions:

  1. Open a new terminal window/tab

  2. Navigate to your project 
  directory:
  cd
  "/Users/dinhquanghien/Documents/Học 
  tập/pre_2"

  3. Start the server:
  python app_chatbot.py

  4. You should see output like:
  INFO:     Started server process
  [xxxxx]
  INFO:     Waiting for application
  startup.
  INFO:     Application startup
  complete.
  INFO:     Uvicorn running on
  http://0.0.0.0:8000 (Press CTRL+C to
  quit)

  5. Open another new terminal to test 
  it:
  # Test health endpoint
  curl -X GET http://127.0.0.1:8000/health


  # Test prediction with Vietnamese 
  symptoms
  curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Tôi bị ho và sốt suốt ba ngày nay"}'


  6. To stop the server: Press Ctrl+C
  in the terminal running the server

  The server will run on
  http://127.0.0.1:8000 and you can
  access:
  - /health - Check server status
  - /predict - Send Vietnamese symptoms
   for disease prediction
   
Mình thấy UI đã chạy, bạn nhập triệu chứng và bot trả về: “Xin lỗi, có lỗi xảy ra khi gọi API”. Điều này nghĩa là frontend kết nối được nhưng backend trả lỗi.

👉 Nguyên nhân phổ biến khi bạn mở index.html bằng Live Server (http://127.0.0.1:5500/
) và backend chạy http://127.0.0.1:8000/
 là: CORS (Cross-Origin Resource Sharing) chưa bật trong FastAPI. Browser chặn request do khác port.
   pip install "fastapi[all]" "uvicorn[standard]" python-multipart
