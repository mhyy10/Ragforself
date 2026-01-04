# 企业级 RAG 知识库系统

这是一个基于 LangChain 构建的企业级 RAG（检索增强生成）知识库系统，专为中小型企业设计。

## 系统架构

- **前端界面**: 简单的 HTML/CSS/JavaScript 界面，支持文档上传和问答功能
- **API 层**: 使用 FastAPI 构建 RESTful API
- **核心处理层**: 文档处理、向量化、检索和生成模块
- **向量数据库**: ChromaDB 存储文档向量
- **语言模型**: 支持 OpenAI GPT 或本地模型

## 功能特性

- 文档上传（支持 PDF, DOCX, TXT, HTML 等格式）
- 语义化文档检索
- 基于上下文的智能问答
- 用户认证和权限管理
- 完整的日志记录
- 监控和健康检查

## 快速开始

### 环境要求

- Python 3.9+
- OpenAI API 密钥（可选，也支持本地模型）

### 安装步骤

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   ```bash
   # 编辑 .env 文件，添加 OpenAI API 密钥
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. 启动服务：
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### API 端点

- `GET /` - 主页
- `GET /health` - 系统健康检查
- `POST /api/documents/upload` - 上传文档
- `POST /api/chat/query` - 问答查询
- `GET /api/documents/status` - 系统状态
- `POST /api/auth/token` - 用户认证

### 使用方式

1. 访问前端界面: `ui/index.html`
2. 上传文档到知识库
3. 提出问题并获得基于文档的智能回答

## 项目结构

```
rag_system/
├── api/                 # API层 - FastAPI应用
├── ui/                  # 前端界面
├── core/                # 核心业务逻辑
│   ├── document_processor.py # 文档处理核心
│   ├── embedding_engine.py   # 嵌入引擎
│   ├── retrieval_engine.py   # 检索引擎
│   └── generation_engine.py  # 生成引擎
├── config/              # 配置文件
├── utils/               # 工具函数
├── docs/                # 文档
└── tests/               # 测试文件
```

## 部署

### Docker 部署

```bash
docker-compose up -d
```

### 生产环境部署

请使用以下命令部署到生产环境：

```bash
python deploy.py prod
```

## 安全说明

- 实现了 JWT 基础的用户认证
- 文件上传有类型限制和大小限制
- 使用了输入验证和清理机制

## 维护

系统包含完整的日志记录机制，日志文件保存在 `logs/` 目录中。

## 扩展性

此系统设计为模块化架构，易于扩展新功能，如：

- 更多文档格式支持
- 高级检索策略
- 多语言支持
- 企业级集成（LDAP, SSO）