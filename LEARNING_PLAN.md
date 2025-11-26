# 📚 Learning Plan: From Data Loading to Model Training

## ✅ What You Already Know
- ✅ Data loads from CSV → DataFrame
- ✅ You understand: `df = loader.load_data(...)`

---

## 🎯 Step-by-Step Learning Path

### **Session 1: Understanding Features and Labels** (30 min)

**Goal:** Learn what features (X) and labels (y) are.

**What happens next in your code:**
```python
X, y = loader.get_features_labels()  # Line 97 in main.py
```

**Simple Explanation:**
- **Features (X)**: Input numbers used to predict
- **Labels (y)**: What you want to predict

**Real Example:** Predicting if email is spam
- Features: word counts, sender, length → `X`
- Label: spam or not spam → `y`

**In your code (lines 116-127):**
```python
X_df = self.df[feature_cols]      # All columns EXCEPT the label
y_series = self.df[label_col_name]  # Only the label column
```

**Action Items:**
1. Read `src/data_loader.py` lines 85-127
2. Try to identify:
   - Which columns become features?
   - Which column is the label?

**Check Your Understanding:**
- [ ] I can explain what features are
- [ ] I can explain what labels are
- [ ] I understand why we separate them

---

### **Session 2: Train/Validation/Test Split** (20 min)

**Goal:** Learn why we split data into 3 parts.

**What happens:**
```python
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
```

**Simple Explanation:**
- **Train**: Model learns from this (like studying)
- **Validation**: Check performance during training (like practice tests)
- **Test**: Final check on unseen data (like final exam)

**Analogy:** 
- Train = Textbook (you study from it)
- Validation = Practice Tests (you check your progress)
- Test = Final Exam (you don't see it until the end)

**Action Items:**
1. Read `src/data_loader.py` lines 136-176
2. Answer: What percentages go to train/val/test?

**Check Your Understanding:**
- [ ] I understand why we need 3 splits
- [ ] I know what each split is used for
- [ ] I can explain the analogy

---

### **Session 3: Data Preprocessing** (30 min)

**Goal:** Learn why we preprocess data.

**What happens:**
```python
preprocessor.preprocess_features(X_train, X_val, X_test, n_features=16)
```

**Simple Explanation:**
- Features are on different scales (some 0-1, some 0-1000)
- We standardize them so big numbers don't dominate
- We reduce dimensions (PCA/selection) to make it manageable

**Analogy:** Converting currencies to common unit before comparing prices.

**Action Items:**
1. Read `src/data_preprocessor.py` lines 28-81
2. What does StandardScaler do?
3. What does PCA do?

**Check Your Understanding:**
- [ ] I understand why we scale features
- [ ] I know what PCA does
- [ ] I understand why we reduce dimensions

---

### **Session 4: Neural Networks Basics** (45 min)

**Goal:** Understand what a model is.

**What is a Neural Network?**
- A function that takes numbers in → returns predictions
- Example: `prediction = model(input_features)`

**Simple Example:**
```python
# Imagine a simple model:
def simple_model(feature1, feature2):
    result = 0.5 * feature1 + 0.3 * feature2 + 1.0
    return "spam" if result > 0.5 else "not spam"
```

Your model is more complex, but same idea: **input numbers → prediction**

**Action Items:**
1. Watch 10-min YouTube video: "Neural Network Explained Simply"
2. Read `src/quantum_transformer.py` lines 220-261 (just the forward function)
3. Try to trace: input → what happens → output

**Check Your Understanding:**
- [ ] I can explain what a neural network does
- [ ] I understand input → model → output flow
- [ ] I know what "forward pass" means

---

### **Session 5: Understanding Your Quantum Transformer** (30 min)

**Goal:** See how components fit together.

**Your Model Structure:**
```
Input Data (X)
    ↓
Embedding Layer (converts to numbers model understands)
    ↓
Transformer Blocks (the "brain" - processes information)
    ↓
Classifier (final prediction)
```

**Action Items:**
1. Read `src/quantum_transformer.py` lines 176-218
2. Don't worry about quantum yet - just see the layers
3. Draw a simple diagram of the flow

**Check Your Understanding:**
- [ ] I can list the main parts of the model
- [ ] I understand what embedding does
- [ ] I know what transformer blocks do

---

### **Session 6: Training Basics** (40 min)

**Goal:** Learn what training means.

**Simple Explanation:**
- **Training** = Showing examples, model learns from mistakes
- **Loss** = How wrong the model is
- **Optimizer** = Adjusts model to reduce errors

**Analogy:** Learning to ride bike by practice and adjusting.

**Your code does this:**
```python
for each batch of data:
    1. Make prediction
    2. Calculate how wrong it is (loss)
    3. Adjust model to be less wrong
    4. Repeat
```

**Action Items:**
1. Read `src/trainer.py` lines 311-404
2. Look for:
   - Where model makes predictions
   - Where loss is calculated
   - Where model gets adjusted

**Check Your Understanding:**
- [ ] I understand what training means
- [ ] I know what loss is
- [ ] I understand what optimizer does

---

### **Session 7: Hyperparameters** (20 min)

**Goal:** Understand what hyperparameters are.

**Simple Explanation:**
- **Hyperparameters** = Settings (not learned by model)
- Examples: learning rate, number of layers, dropout

**In your code:**
```python
embed_dim=64           # How big the model is
n_transformer_layers=3 # How many layers
learning_rate=0.001    # How fast it learns
```

**Action Items:**
1. In `main.py` line 158, what hyperparameters are being optimized?
2. Read `src/automl_optimizer.py` lines 183-196
3. Try changing one number and see what happens

**Check Your Understanding:**
- [ ] I can explain what hyperparameters are
- [ ] I know the difference between parameters and hyperparameters
- [ ] I understand why we optimize them

---

### **Session 8: Quantum Attention** (Advanced - Later)

**Goal:** Understand the quantum part (optional for now).

**For now, just know:**
- "Quantum attention is like regular attention, but uses quantum circuits instead of simple math"
- You can skip this until you understand classical parts

**Action Items:**
1. Come back to this after Sessions 1-7
2. Read `src/quantum_attention.py` when ready

---

## 📅 Daily Practice Plan

### **Week 1: Data Processing**
- [ ] Session 1: Features and Labels
- [ ] Session 2: Train/Val/Test Split
- [ ] Session 3: Data Preprocessing

**Focus:** Understanding how data flows through your pipeline

### **Week 2: Model Basics**
- [ ] Session 4: Neural Networks Basics
- [ ] Session 5: Your Quantum Transformer Structure

**Focus:** Understanding what your model does

### **Week 3: Training**
- [ ] Session 6: Training Basics
- [ ] Session 7: Hyperparameters

**Focus:** Understanding how model learns

### **Week 4: Advanced (Optional)**
- [ ] Session 8: Quantum Attention

**Focus:** Understanding the quantum component

---

## 📖 How to Study Each Session

1. **Read** the code section mentioned
2. **Run** a small test if possible
3. **Write** 2-3 sentences about what you learned
4. **Try** to explain it to yourself out loud

---

## 🎓 Resources

### **Free Courses:**
- **Kaggle Learn**: "Intro to Machine Learning" (2 hours)
- **YouTube**: "3Blue1Brown - Neural Networks Playlist"

### **Practice:**
- Try changing numbers in your code
- Print intermediate values to see what they look like

---

## ✅ Questions to Check Your Understanding

After each session, answer:
1. What does this step do?
2. Why do we need this step?
3. What happens if we skip it?

---

## 🚀 Getting Started

**Start with Session 1** (Features and Labels) - should take ~30 minutes.

**When ready for Session 2, come back and continue!**

---

## 📝 Notes Section

Use this space to write your own notes as you learn:

### Session 1 Notes:
- 

### Session 2 Notes:
- 

### Session 3 Notes:
- 

### Session 4 Notes:
- 

### Session 5 Notes:
- 

### Session 6 Notes:
- 

### Session 7 Notes:
- 

### Session 8 Notes:
- 

---

## 💡 Tips

- **Don't rush** - Take your time with each session
- **Ask questions** - If something doesn't make sense, ask!
- **Practice** - Try modifying code to see what happens
- **Review** - Come back to previous sessions if needed

---

**Remember:** Everyone starts here. You're doing great! 🎉






