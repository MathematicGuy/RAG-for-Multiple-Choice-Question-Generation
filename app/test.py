import os
print(os.environ.get('HF_API_KEY'))
print(os.environ.get('TOGETHER_KEY'))

# import json
# batch = {
#     '1': {
#         'câu hỏi': 'Theo nội dung đã cho, một trong những nhược điểm của việc fine‑tuning mô hình là gì?',
#         'lựa chọn': {
#             'a': 'Yêu cầu ít tài nguyên tính toán',
#             'b': 'Đòi hỏi tài nguyên tính toán lớn',
#             'c': 'Dễ dàng áp dụng cho mọi tác vụ',
#             'd': 'Không cần dữ liệu gán nhãn'
#         },
#         'đáp án': 'Đòi hỏi tài nguyên tính toán lớn'
#     },
#     '2': {
#         'câu hỏi': 'Theo công thức (39) trong tài liệu, trọng số được tính như thế nào bằng cách kết hợp lồi hai thành phần similarity và saliency với tham số α?',
#         'lựa chọn': {
#             'a': '(1−α)×w similarity×ICF +α×w saliency',
#             'b': 'α×w similarity×ICF +(1−α)×w saliency',
#             'c': 'w similarity + w saliency',
#             'd': '(1+α)×w similarity×ICF - α×w saliency'
#         },
#         'đáp án': '(1−α)×w similarity×ICF +α×w saliency'
#     },
#     '3': {
#         'câu hỏi': 'Trong phương pháp Explainable AI (XAI) được mô tả, phương pháp nào được sử dụng để xác định các từ khóa quan trọng nhất trong tin nhắn?',
#         'lựa chọn': {
#             'a': 'Masking-based Saliency',
#             'b': 'Synonym Replacement',
#             'c': 'Hard Ham Generation',
#             'd': 'Alpha parameter'
#         },
#         'đáp án': 'Masking-based Saliency'
#     },
#     '4': {
#         'câu hỏi': 'Kiến trúc của mô hình BERT-base bao gồm những thành phần nào sau đây?',
#         'lựa chọn': {
#             'a': '12 lớp encoder Transformer, mỗi lớp có 768 chiều ẩn và 12 head attention, tổng khoảng 110 triệu tham số.',
#             'b': '24 lớp encoder Transformer, mỗi lớp có 1024 chiều ẩn và 16 head attention, tổng khoảng 340 triệu tham số.',
#             'c': '12 lớp decoder Transformer, mỗi lớp có 768 chiều ẩn và 12 head attention, tổng khoảng 110 triệu tham số.',
#             'd': '6 lớp encoder Transformer, mỗi lớp có 512 chiều ẩn và 8 head attention, tổng khoảng 45 triệu tham số.'
#         },
#         'đáp án': '12 lớp encoder Transformer, mỗi lớp có 768 chiều ẩn và 12 head attention, tổng khoảng 110 triệu tham số.'
#     },
#     '5': {
#         'câu hỏi': 'Theo quy trình được mô tả, phương pháp nào được sử dụng để kết hợp các cụm spam/ham với câu "base"?',
#         'lựa chọn': {
#             'a': 'Chỉ sử dụng dạng base + insert',
#             'b': 'Chỉ sử dụng dạng insert + base',
#             'c': 'Cả hai dạng base + insert và insert + base',
#             'd': 'Không sử dụng bất kỳ dạng nào'
#         },
#         'đáp án': 'Cả hai dạng base + insert và insert + base'
#     },
#     '6': {
#         'câu hỏi': 'Trong phương pháp kết hợp điểm số của mô hình, trọng số được gán cho điểm ngữ nghĩa BERT là bao nhiêu?',
#         'lựa chọn': {
#             'a': '30%',
#             'b': '50%',
#             'c': '70%',
#             'd': '90%'
#         },
#         'đáp án': '70%'
#     },
#     '7': {
#         'câu hỏi': 'Trong các tính chất của Inverse Class Frequency (ICF) được nêu, tính chất nào khẳng định rằng tổng của ICF(c) nhân với số mẫu của lớp c trên toàn bộ các lớp bằng 1?',
#         'lựa chọn': {
#             'a': 'Tính đơn điệu',
#             'b': 'Chuẩn hóa',
#             'c': 'Hiệu chỉnh bias',
#             'd': 'Không có tính chất nào như vậy'
#         },
#         'đáp án': 'Chuẩn hóa'
#     },
#     '8': {
#         'câu hỏi': 'Trong tài liệu, loại chỉ mục FAISS nào được cho là cho kết quả tương đương với độ tương đồng cosine?',
#         'lựa chọn': {
#             'a': 'IndexFlatL2',
#             'b': 'IndexFlatIP',
#             'c': 'IndexIVFFlat',
#             'd': 'IndexHNSW'
#         },
#         'đáp án': 'IndexFlatIP'
#     },
#     '9': {
#         'câu hỏi': 'Trong nội dung, kỹ thuật nào được sử dụng để thay thế các từ “nhạy cảm” trong spam bằng từ đồng nghĩa?',
#         'lựa chọn': {
#             'a': 'Thay thế từ đồng nghĩa (Synonym Replacement)',
#             'b': 'Tăng cường dữ liệu (Data Augmentation)',
#             'c': 'Học bán giám sát (Semi-supervised learning)',
#             'd': 'Phân cụm embedding (Embedding clustering)'
#         },
#         'đáp án': 'Thay thế từ đồng nghĩa (Synonym Replacement)'
#     },
#     '10': {
#         'câu hỏi': 'Trong quy trình Feed‑Forward Network (FFN) được mô tả, ma trận trọng số W có kích thước nào?',
#         'lựa chọn': {
#             'a': '3072 × 768',
#             'b': '768 × 3072',
#             'c': '1024 × 768',
#             'd': '3072 × 1024'
#         },
#         'đáp án': '3072 × 768'
#     }
# }


# with open('output.json', 'w', encoding='utf-8') as f:
#     json.dump(batch, f, indent=3)