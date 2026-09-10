"""
sua_ban_dich_cache.py — Áp fix tay cho các bản dịch sai trong cache
derived/mo_rong_truy_van.json, đúng theo cơ chế "SỬA TAY ĐƯỢC" đã thiết kế
sẵn trong dich_bang_marian() (xem docstring của hàm đó).

2 cụm phát hiện dịch sai nghiêm trọng (không phải lỗi nhỏ, mà lệch hẳn
sang khái niệm khác — ảnh hưởng trực tiếp tới CLIP-L retrieval):

  - "cây xả" (cây sả/lá sả = lemongrass) -> bị dịch thành "irrigator"
    (máy tưới tiêu?!). Rất có thể do lỗi chính tả "xả" thay vì "sả" khiến
    Marian đoán nhầm nghĩa.
  - "dồi trường" (dồi lòng heo, món ăn) -> bị dịch thành "Hot pink",
    "Fatty fields", "Egg bran" — vô nghĩa, lặp lại ở nhiều câu.

Chạy 1 lần:
    python -m scripts.sua_ban_dich_cache

An toàn để chạy nhiều lần (idempotent) — chỉ ghi đè đúng các key liệt kê
trong OVERRIDES bên dưới, không đụng gì tới các bản dịch khác.

Nếu tìm thêm bản dịch sai khác sau này, chỉ cần thêm entry vào OVERRIDES,
không cần sửa gì khác trong script.
"""
from __future__ import annotations

from aic2026.query_expand import BoNhoDem

# {nguồn: {câu tiếng Việt gốc (đúng key, kể cả tiền tố "(1) " nếu có): [cụm tiếng Anh mới]}}
OVERRIDES: dict[str, dict[str, list[str]]] = {
    "marian": {
        "(1) Dao vừa chạm vào cây xả": [
            "(1) the knife has just touched the lemongrass stalk.",
            "a photo of (1) the knife has just touched the lemongrass stalk.",
        ],
        "(2) Cây xả bị cắt rời ra": [
            "(2) the lemongrass stalk is cut off.",
            "a photo of (2) the lemongrass stalk is cut off.",
        ],
        "Dao vừa chạm vào cây xả": [
            "The knife just touched the lemongrass stalk.",
            "a photo of The knife just touched the lemongrass stalk.",
        ],
        "dồi trường": [
            "Pork intestine sausage.",
            "a photo of Pork intestine sausage.",
        ],
        "dồi trường màu trắng": [
            "White pork intestine sausage.",
            "a photo of White pork intestine sausage.",
        ],
        "dồi trường trắng": [
            "White pork intestine sausage.",
            "a photo of White pork intestine sausage.",
        ],
        "dồi trường trắng xào": [
            "Stir-fried white pork intestine sausage.",
            "a photo of Stir-fried white pork intestine sausage.",
        ],
        "dồi trường xào bông hẹ": [
            "Stir-fried pork intestine sausage with chives.",
            "a photo of Stir-fried pork intestine sausage with chives.",
        ],
        "dồi trường, hẹ": [
            "Pork intestine sausage, chives.",
            "a photo of Pork intestine sausage, chives.",
        ],
        "dồi trường, rau xanh": [
            "Pork intestine sausage, green vegetables.",
            "a photo of Pork intestine sausage, green vegetables.",
        ],
        "món dồi trường màu trắng và rau xanh": [
            "White pork intestine sausage and green vegetables.",
            "a photo of White pork intestine sausage and green vegetables.",
        ],
        "món dồi trường với hẹ": [
            "Pork intestine sausage with chives.",
            "a photo of Pork intestine sausage with chives.",
        ],
    },
    "tu_dien": {
        "(1) Dao vừa chạm vào cây xả": [
            "knife lemongrass plant",
            "a photo of knife lemongrass plant",
        ],
        "(2) Cây xả bị cắt rời ra": [
            "lemongrass plant cutting",
            "a photo of lemongrass plant cutting",
        ],
        "Cây xả bị cắt rời ra": [
            "lemongrass plant cutting",
            "a photo of lemongrass plant cutting",
        ],
        "Dao vừa chạm vào cây xả": [
            "knife lemongrass plant",
            "a photo of knife lemongrass plant",
        ],
    },
}


def main() -> None:
    bo_nho = BoNhoDem()

    n_sua = 0
    for nguon, cac_cau in OVERRIDES.items():
        for cau, cum_moi in cac_cau.items():
            cu = bo_nho.lay(nguon, cau)
            bo_nho.dat(nguon, cau, cum_moi)
            trang_thai = "SỬA" if cu is not None else "THÊM MỚI"
            print(f"[{nguon}] {trang_thai}: {cau!r}")
            print(f"    cũ: {cu}")
            print(f"    mới: {cum_moi}")
            n_sua += 1

    bo_nho.ghi()
    print(f"\nĐã áp {n_sua} bản sửa vào {bo_nho.duong_dan}")


if __name__ == "__main__":
    main()