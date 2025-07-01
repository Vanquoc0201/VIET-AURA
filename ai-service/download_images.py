# download_images.py
from icrawler.builtin import GoogleImageCrawler

# Danh sách các từ khóa để tải ảnh
categories = {
    "non_la": "nón lá việt nam",
    "ao_dai": "áo dài việt nam",
    "pho_bo": "phở bò việt nam",
    "banh_mi": "bánh mì việt nam"
}

limit = 50  # Số ảnh cho mỗi loại

for label, keyword in categories.items():
    print(f"🔽 Tải ảnh: '{keyword}' → thư mục: dataset/{label}")
    crawler = GoogleImageCrawler(storage={"root_dir": f"dataset/{label}"})
    crawler.crawl(keyword=keyword, max_num=limit)

print("✅ Đã tải xong toàn bộ ảnh.")
