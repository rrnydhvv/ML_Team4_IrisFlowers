# K-Nearest Neighbors (KNN)

Tài liệu mô tả chi tiết cách KNN được cài đặt trong dự án, bao gồm việc tính khoảng cách Euclid, chọn láng giềng, gán trọng số và quyết định nhãn.

## 3.1 Cơ chế dự đoán (chi tiết thuật toán)

### 3.1.1 Tính khoảng cách Euclid
Với mỗi mẫu hoa cần dự đoán, thuật toán tính khoảng cách Euclid đến toàn bộ các mẫu trong tập huấn luyện dựa trên 4 đặc trưng hình thái. Khoảng cách Euclid giữa hai vectơ đặc trưng $\mathbf{x}$ và $\mathbf{y}$ với 4 chiều được tính theo công thức:

$$
 d(\mathbf{x},\mathbf{y}) = \sqrt{\sum_{j=1}^{4} (x_j - y_j)^2}
$$

Trong triển khai, khoảng cách này được tính thủ công bằng vòng lặp `for`: cộng dồn tổng bình phương sai khác của từng thuộc tính rồi lấy căn bậc hai để thu được khoảng cách cuối cùng. Việc tính bằng vòng lặp giúp minh bạch cho bài thực nghiệm và dễ kiểm soát khi dataset nhỏ như Iris.

**Vì sao chọn Euclid?**
- Dữ liệu Iris gồm các đặc trưng liên tục (độ dài/chiều rộng), có cùng đơn vị (cm), nên Euclid là phép đo trực quan và phù hợp để phản ánh sự khác biệt hình thái.
- Euclid đơn giản, dễ hiểu và tính toán nhanh cho kích thước chiều nhỏ (4 chiều).

### 3.1.2 Sắp xếp và chọn K láng giềng
Sau khi tính được khoảng cách từ mẫu cần dự đoán đến tất cả mẫu huấn luyện, các mẫu huấn luyện được sắp xếp theo thứ tự khoảng cách tăng dần. Từ danh sách này, $k$ mẫu có khoảng cách nhỏ nhất được chọn làm $k$ láng giềng gần nhất của mẫu cần dự đoán.

Việc chọn giá trị $k$ phù hợp là quan trọng: $k$ quá nhỏ dễ gây nhiễu (overfit), $k$ quá lớn có thể làm mờ ranh giới phân lớp (underfit). Do đó thường sử dụng K-Fold Cross-Validation để chọn $k$ tối ưu (xem phần 3.2).

### Gán trọng số theo khoảng cách
Mỗi láng giềng trong $k$ láng giềng gần nhất được gán một trọng số tỉ lệ nghịch với khoảng cách đến mẫu cần dự đoán. Các mẫu nằm càng gần sẽ có trọng số càng lớn, trong khi các mẫu ở xa đóng góp ít hơn. Một hằng số rất nhỏ được thêm vào để tránh trường hợp chia cho 0.

Biểu thức gán trọng số theo khoảng cách được sử dụng là:

$w_i = \\dfrac{1}{d_i + \\varepsilon}$

trong đó $d_i$ là khoảng cách Euclid từ mẫu cần dự đoán đến láng giềng thứ $i$ trong số $k$ láng giềng gần nhất, còn $\\varepsilon$ là một hằng số rất nhỏ (ví dụ $1\\times10^{-9}$) để tránh chia cho 0 khi khoảng cách bằng 0. Theo biểu thức này, láng giềng có khoảng cách càng nhỏ sẽ được gán trọng số càng lớn và ảnh hưởng lớn hơn đến kết quả phân loại.

### Quyết định nhãn bằng Voting
Sau khi có $k$ láng giềng và trọng số tương ứng, thuật toán tính tổng trọng số cho mỗi nhãn (label). Nhãn có tổng trọng số lớn nhất sẽ được chọn làm nhãn dự đoán cho mẫu mới (weighted voting). Cụ thể:

- Tổng trọng số của nhãn $c$ là $S_c = \\sum_{i\\in N_c} w_i$, trong đó $N_c$ là tập các láng giềng có nhãn $c$.
- Dự đoán nhãn $\\hat{y} = \\arg\\max_c S_c$.

## 3.2 Tối ưu tham số $k$ bằng K-Fold Cross-Validation
Để chọn giá trị $k$ tốt nhất, dùng K-Fold Cross-Validation (mặc định thường là 5-fold):

- Chia dữ liệu huấn luyện thành $K$ folds; với mỗi giá trị $k$ thử nghiệm, lặp qua các fold, dùng $K-1$ fold để train (lưu mẫu) và fold còn lại để đánh giá accuracy trung bình.
- Chọn $k$ có accuracy trung bình cao nhất hoặc ổn định nhất trên các fold.

**Vì sao dùng K-Fold?**
- K-Fold giúp ước lượng hiệu năng tổng quát hoá (generalization) của một giá trị $k$ trên các phần khác nhau của dữ liệu thay vì chỉ một phép chia train/test đơn lẻ.
- Với dataset nhỏ như Iris, K-Fold (ví dụ 5-fold) tận dụng dữ liệu hiệu quả hơn và giảm phương sai của ước lượng so với một split đơn.

## 3.3 Tóm tắt lý do lựa chọn chính

- Euclid: phù hợp cho các đặc trưng liên tục cùng đơn vị, trực quan và nhanh cho chiều nhỏ.
- K-Fold: chọn $k$ tối ưu và đánh giá khả năng tổng quát hoá, đặc biệt hữu ích với bộ dữ liệu nhỏ.
- Weighting: gán trọng số theo nghịch đảo khoảng cách làm tăng ảnh hưởng của các láng giềng gần, giảm thiểu tác động của láng giềng xa và giúp giải quyết tình huống hoà (ties) trong voting.

## 3.4 Ghi chú triển khai

- Trong mã nguồn, khoảng cách được tính bằng vòng lặp `for` để minh bạch và dễ kiểm chứng.
- Khi dùng dữ liệu có thang đo khác nhau, cần chuẩn hoá (standardize/min-max) trước khi tính Euclid.
- $\\varepsilon$ nên là một số rất nhỏ như $1\\times10^{-9}$ để tránh chia cho 0 mà không làm thay đổi thứ tự ưu tiên các láng giềng.

---

Tệp `src/models/KNN.py` chứa cài đặt chi tiết; phần này mô tả lý thuyết và các quyết định thiết kế quan trọng cho KNN trong dự án.
