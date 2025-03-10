
### **EANNS – Enhanced Approximate Nearest Neighbor Search**  
🚀 **High-Performance Vector Database**  


EANNS (**Enhanced Approximate Nearest Neighbor Search**) is a **high-performance, open-source vector database** built from scratch in **C++**, designed for **fast and scalable nearest neighbor search**. Inspired by **FAISS**, EANNS introduces **hybrid storage (RAM & Disk), real-time updates, and metadata-aware filtering**, making it suitable for AI-driven search applications.  

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

