


![EANNS Logo](https://github.com/AmanSwar/EANNS/blob/master/images/eannslogo.png)
# **EANNS – Enhanced Approximate Nearest Neighbor Search**  

EANNS is a **high-performance, hybrid vector database** designed for **real-time, scalable, and metadata-aware vector search**. Unlike traditional ANN solutions like FAISS, Milvus, or Weaviate, EANNS is optimized for **both speed and persistence**, leveraging **RAM for ultra-fast queries** and **disk storage for long-term scalability**.  

#### **Key Features:**  
✅ **Hybrid Storage** → RAM (fast retrieval) + Disk (persistent storage)  
✅ **Hybrid Search** → Combines vector similarity search with structured metadata filtering  
✅ **Real-Time Updates** → No need for full reindexing like FAISS; supports dynamic indexing  
✅ **Optimized with CUDA & OpenMP** → Extreme speed via parallel computing  
✅ **Redis-Style Simplicity** → Minimal setup, easy-to-use API, open-source scalability  

Built in **C++ with CUDA and OpenMP**, EANNS is designed for **high-performance AI, search, and recommendation systems**, offering **unmatched efficiency and flexibility** compared to existing vector search solutions. 🚀

### 🔥 **Current Status: Under Development**  
EANNS is in **active development**, with key components being implemented iteratively. The first release will focus on:  
 **Core vector storage** (RAM/Disk hybrid)  
 **Brute-force search (Flat Index)**  
 **Efficient indexing with SIMD & parallelism**  
 **IVF, HNSW, and PQ-based search (WIP)**  
 **Real-time indexing & metadata filtering (WIP)**  
 **CUDA-accelerated ANN search (Coming soon)**  

---

## ⚡ **Key Features (Planned)**  
### **1. Flexible Storage & Search**  
- **Multiple Index Types**:  
  - **SpaceFlat** → Brute-force search (like IndexFlatL2).  
  - **SpaceCluster** → Cluster-based search (like IndexIVF).  
  - **SpaceGraph** → Graph-based search (like IndexHNSW).  
  - **SpaceQuantize** → Compressed search (like IndexPQ).  
- **Hybrid Storage**: RAM (fast access) & Disk (persistent storage).  

### **2. Optimized for Speed & Scale**  
- **CUDA-accelerated search** (for GPU compute).  
- **OpenMP for multi-threaded query execution**.  
- **SIMD-powered vectorized computations**.  

### **3. Metadata-Aware Hybrid Search**  
- **Supports metadata filtering alongside vector similarity**.  
- **Key-value store for fast lookup & hybrid queries**.  

---

## 🛠 **Tech Stack**  
- **Language**: C++  
- **Parallelism**: CUDA, OpenMP, SIMD  
- **Math Libraries**: Eigen  
- **Storage**: Custom memory manager (RAM/Disk)  

---

## 📌 **Development Roadmap**  
### **MVP (v0.1 Release) – Q2 2025**  
✔️ Implement **SpaceFlat (brute-force search)**  
✔️ **Basic query engine** (single-node)  
✔️ **SIMD-optimized computations**  
✔️ **Basic Python bindings**  

### **v0.2+ (Scaling & Optimization)**  
🚧 Add **SpaceCluster (IVF)**  
🚧 Add **SpaceGraph (HNSW)**  
🚧 **CUDA-based vector search**  
🚧 **Persistent disk-based storage**  

---

## 🤝 **Contributions & Feedback**  
EANNS is an **open-source project**, and we welcome feedback, contributions, and collaborations! Join us in building a **fast, scalable, and open alternative** to proprietary vector search engines.  

📢 **Star this repo** to get updates as development progresses! ⭐  

---

### 🚀 **Stay Tuned!**  
This project is cooking! 🔥 Expect **initial benchmarks, API designs, and first releases** soon. Follow the repo for updates.  

---

### **📂 Installation (Coming Soon)**  
Instructions for building and using EANNS will be provided in the first release.  

---

## **📜 License**  
EANNS will be released under an **open-source license** soon. Stay tuned!  

---

### **📧 Contact & Updates**  
For discussions and updates, feel free to connect via GitHub Issues or Discussions.  

