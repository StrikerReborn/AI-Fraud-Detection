## 📊 Live Prediction Results

The system was tested with three different transaction-risk scenarios through the Streamlit interface.

### 🟢 Low-Risk Transaction

The model identified the transaction as low risk and recommended approval.

![Low Risk Prediction](images/low-risk.png)

**Result:** LOW → APPROVE

---

### 🟡 Medium-Risk Transaction

The model identified elevated fraud probability and recommended manual review.

![Medium Risk Prediction](images/medium-risk.png)

**Result:** MEDIUM → REVIEW

---

### 🔴 High-Risk Transaction

The model identified a very high fraud probability and recommended blocking the transaction.

![High Risk Prediction](images/high-risk.png)

**Result:** HIGH → BLOCK