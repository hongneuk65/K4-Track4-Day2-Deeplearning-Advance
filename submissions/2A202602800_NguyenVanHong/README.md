# Lab Day 2 — Nguyễn Văn Hồng

MSSV: 2A202602800

Notebook Colab: https://colab.research.google.com/drive/1hHWr3tceI_zQFmJ2MLObZo4HFrgQvVIu

Link trên là notebook của lần thực nghiệm đã chạy. Code train/inference trong
bài nộp giữ nguyên bản đã có trên GitHub tại commit `bfd8765`. Notebook trong
`code/` chỉ bổ sung phần tạo sản phẩm Bước 5 (Summary và biểu đồ backbone),
không thay đổi cấu hình hay phép tính train/test. Phần bổ sung này đã được
chạy riêng trên CPU từ Excel có sẵn; không tuyên bố đã chạy lại toàn bộ
notebook trên Colab. Giữ link cũ làm bằng chứng thực nghiệm.

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

Checkpoint và ảnh dataset không được commit lên GitHub.
Để train lại, dùng notebook và cùng cấu hình đã ghi.
Nếu tiếp tục phiên bị ngắt, cần giữ checkpoint trên Drive.


## Kiểm tra nhanh trên CPU

Từ thư mục code: `python test_helpers.py -v`.
Không cần dataset hoặc GPU. Các helper bổ sung không phải thí nghiệm mới.
EMA là helper độc lập, chưa tích hợp vào run(); các thí nghiệm
đã nộp không dùng EMA. Đây là hạn chế của phần hoàn thiện starter.
CLI hỗ trợ các field
Config đã dùng. run() giữ save_test_predictions=False; test được chạy riêng
ở Bước 4 của notebook để áp dụng suy luận đã chốt.

Notebook và code/*.py đã đồng bộ các cell %%writefile. Dùng notebook hiện
trong thư mục bài nộp này để tái lập. Giữ bản Colab trên Drive làm bằng chứng
thực nghiệm; nếu chạy bản hoàn thiện, lưu thành notebook riêng và thư mục mới.
Các phiên bản từng lần train phải lấy từ logs/*/seed*/environment.json;
requirements-freeze.txt chỉ là snapshot của một phiên, không đại diện mọi phiên.

## Cấu trúc bài nộp GitHub

Bài nằm tại `submissions/2A202602800_NguyenVanHong/`, theo README mục 5 của repo.

```text
submissions/2A202602800_NguyenVanHong/
├── README.md
├── results.xlsx
├── report.md
├── curves/
├── predictions/
└── code/
    ├── lab_day2.ipynb
    └── *.py
```

Kèm `logs/`, `final_manifest.json`, `split_check.json`,
`requirements-freeze.txt` và `F01_seed0_errors.csv` để đối chiếu kết quả.
`eval_out/` là đầu ra đánh giá có thể tạo lại từ predictions, không cần commit.

Trên Colab, thư mục làm việc vẫn là
`/content/drive/MyDrive/2A202602800_NguyenVanHong`; đây là nơi lưu kết quả chạy,
khác với vị trí bài nộp trong repo. Mở notebook trong `code/` bằng Colab,
bật GPU và chạy các cell theo thứ tự trên để tái lập.

Từ thư mục gốc repo, kiểm tra helper bằng:
`python submissions/2A202602800_NguyenVanHong/code/test_helpers.py -v`.
Nộp link GitHub đến thư mục bài làm này theo hướng dẫn của giảng viên.

Để tạo lại Summary và biểu đồ backbone từ Excel đã có, không cần GPU:
`python code/submission_artifacts.py .` (chạy trong thư mục bài nộp).
Summary xếp top 10 cấu hình B/T/I theo macro-F1 val seed 0, gộp các dòng
trùng B03/T00 và T08/I00. I03 được tô nổi bật vì là suy luận đã chọn;
I02 đứng đầu về F1 val. Độ trễ tham khảo và độ trễ đo trực tiếp được ghi rõ.
