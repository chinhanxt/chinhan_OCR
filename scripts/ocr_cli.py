#!/usr/bin/env python3
import sys
import os
import subprocess
import tempfile
import requests
import argparse

API_URL = os.getenv("OCR_API_URL", "http://localhost:3000/v1/ocr")

def get_image_from_clipboard():
    """Extract image data from system clipboard using wl-paste or xclip."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()

    # Try wl-paste (Wayland)
    try:
        res = subprocess.run(["wl-paste", "--type", "image/png"], stdout=open(tmp_path, "wb"), stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.getsize(tmp_path) > 0:
            return tmp_path
    except Exception:
        pass

    # Try xclip (X11)
    try:
        res = subprocess.run(["xclip", "-selection", "clipboard", "-t", "image/png", "-o"], stdout=open(tmp_path, "wb"), stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.getsize(tmp_path) > 0:
            return tmp_path
    except Exception:
        pass

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return None

def send_ocr_request(image_path, mode="gundam"):
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Tệp '{image_path}' không tồn tại.")
        sys.exit(1)

    is_pdf = image_path.lower().endswith(".pdf")
    mime_type = "application/pdf" if is_pdf else "image/png"
    print(f"⏳ Đang gửi {'PDF' if is_pdf else 'ảnh'} '{os.path.basename(image_path)}' tới Unlimited-OCR Server ({mode} mode)...")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, mime_type)}
            params = {"mode": mode}
            response = requests.post(API_URL, files=files, params=params, timeout=300)

        if response.status_code == 200:
            data = response.json()
            parsed_text = data.get("parsed_text", "").strip()
            print("\n" + "="*50)
            print("📝 KẾT QUẢ TRÍCH XUẤT OCR:")
            print("="*50)
            print(parsed_text if parsed_text else "(Không tìm thấy văn bản)")
            print("="*50 + "\n")
            return parsed_text
        else:
            print(f"❌ Lỗi API ({response.status_code}): {response.text}")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("❌ Không thể kết nối tới OCR Server tại http://localhost:8000.")
        print("💡 Hãy chắc chắn bạn đã khởi chạy Docker container: 'sudo docker compose up -d'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Unlimited-OCR Terminal CLI Tool")
    parser.add_argument("image_path", nargs="?", help="Đường dẫn tệp ảnh hoặc PDF. Nếu bỏ trống, script sẽ tự động lấy ảnh từ Clipboard.")
    parser.add_argument("-m", "--mode", choices=["gundam", "base"], default="gundam", help="Chế độ xử lý: 'gundam' (mặc định - crop chi tiết) hoặc 'base' (full trang)")
    parser.add_argument("-o", "--output", help="Lưu kết quả ra file .txt")

    args = parser.parse_args()

    image_file = args.image_path
    is_temp = False

    if not image_file:
        print("📋 Đang kiểm tra ảnh chụp màn hình trong Clipboard...")
        image_file = get_image_from_clipboard()
        if not image_file:
            print("❌ Không tìm thấy hình ảnh nào trong Clipboard!")
            print("💡 Mẹo: Hãy chụp màn hình (PrintScreen / Ctrl+Shift+PrintScreen hoặc dùng Snipping tool) rồi chạy lại lệnh.")
            sys.exit(1)
        print("✅ Đã tìm thấy ảnh chụp màn hình!")
        is_temp = True

    try:
        result_text = send_ocr_request(image_file, mode=args.mode)
        
        if args.output and result_text:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result_text)
            print(f"💾 Đã lưu kết quả vào file: {args.output}")

    finally:
        if is_temp and os.path.exists(image_file):
            os.remove(image_file)

if __name__ == "__main__":
    main()
