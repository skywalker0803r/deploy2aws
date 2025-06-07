FROM python:3.9-slim

# 安裝必要的系統依賴
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製程式碼與需求
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製剩下的檔案（含 app.py、utils/、model.pth 等）
COPY . .

# 啟動 Flask
CMD ["python", "app.py"]
