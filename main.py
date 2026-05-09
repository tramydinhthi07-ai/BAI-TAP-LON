import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
# 1. Đọc file dữ liệu kế toán
df = pd.read_csv('financial_anomaly_data.csv')
# 2. Xử lý dữ liệu: Chuyển chữ thành số để AI hiểu được
le = LabelEncoder()
df['TransactionType_N'] = le.fit_transform(df['TransactionType'])
df['Location_N'] = le.fit_transform(df['Location'])
# 3. Chọn các đặc trưng để soi: Số tiền, Loại giao dịch, Địa điểm
feature = ['Amount', 'TransactionType_N', 'Location_N']
X = df[feature]
# 4. Khởi tạo mô hình Isolation Forest
# Giả định tỷ lệ gian lận là 1% (0.01)
model = IsolationForest(contamination=0.01, random_state=42)
# 5. Huấn luyện mô hình và tìm điểmm bất thường
df['Anomaly'] = model.fit_predict(X)
# 6. Lọc ra danh sách các giao dịch nghi ngờ (đánh dấu -1)
kq_bat_thuong = df[df['Anomaly'] == -1]
print("--- KẾT QUẢ PHÂN TÍCH GIAN LẬN ---")
print(F"Phát hiện {len(kq_bat_thuong)} giao dịch bất thường.")
print(kq_bat_thuong[['Timestamp' , 'Amount', 'TransactionType', 'Location']].head())
# 7. Xuất toàn bộ 2163 dòng bất thường ra file Excel xịn
kq_bat_thuong.to_excel('Ket_Qua_Gian_Lan_Chuan.xlsx', index=False)
print("Đã tạo file Ket_Qua_Gian_Lan_Chuan.xlsx trùng khớp 100% với kết quả trên!")