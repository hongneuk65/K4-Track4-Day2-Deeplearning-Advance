# Lab Day 2 — Nguyễn Văn Hồng

MSSV: 2A202602800

Notebook Colab: https://colab.research.google.com/drive/1hHWr3tceI_zQFmJ2MLObZo4HFrgQvVIu

## Thứ tự chạy

1. Mount Drive, clone repo, cài môi trường, tải dữ liệu.
2. Viết module bằng các cell %%writefile.
3. Bước 0: kiểm tra split, EDA và pipeline.
4. Bước 1: so sánh B01–B05.
5. Bước 2: T00–T07, rồi kết hợp T08.
6. Bước 3: I00–I04 trên val.
7. Bước 4: chốt manifest, train 3 seed cho F01 và T00,
   test một lượt/model, score, latency và grade.
8. Bước 5: Excel, confusion, phân tích lỗi, báo cáo, đóng gói.

## Cấu hình cuối

ConvNeXt-Tiny + CutMix + label smoothing 0.1.
Suy luận 1-view FP32, temperature khớp riêng trên val mỗi seed.
Baseline: ConvNeXt-Tiny, basic augmentation, CE, 1-view.

Seed sàng lọc: 0. Seed chung kết và baseline: 0,1,2.
Chi tiết cấu hình, GPU và phiên bản nằm trong logs/.
requirements-freeze.txt là bản thư viện đã ghi;
environment.json của từng lần chạy là nguồn đối chiếu trực tiếp.

## Tái lập và đánh giá

Dùng nguyên fold 0. Chọn cấu hình/checkpoint bằng val.
Không dùng test để tinh chỉnh.
code/eval.py được chép nguyên bản từ repo.
File dự đoán trong predictions/ cho phép tính lại các chỉ số.

Checkpoint và ảnh dataset không nằm trong ZIP.
Để train lại, dùng notebook và cùng cấu hình đã ghi.
Nếu tiếp tục phiên bị ngắt, cần giữ checkpoint trên Drive.


## Kiểm tra nhanh trên CPU

Từ thư mục code: `python test_helpers.py -v`.
Không cần dataset hoặc GPU. Các helper bổ sung không phải thí nghiệm mới.
EMA là helper độc lập, chưa tích hợp vào recipe run(); CLI hỗ trợ các field
Config đã dùng. run() giữ save_test_predictions=False; test được chạy riêng
ở Bước 4 của notebook để áp dụng suy luận đã chốt.

Notebook và code/*.py đã đồng bộ các cell %%writefile. Dùng notebook hiện
trong gói này; bản Colab trên Drive cần upload/cập nhật nếu muốn cùng nội dung.
Các phiên bản từng lần train phải lấy từ logs/*/seed*/environment.json;
requirements-freeze.txt chỉ là snapshot của một phiên, không đại diện mọi phiên.
