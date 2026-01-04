rag_system/
├── api/                 # API层 - FastAPI应用
│   ├── __init__.py
│   ├── main.py          # 主API应用
│   ├── routes/          # API路由
│   │   ├── __init__.py
│   │   ├── documents.py # 文档管理路由
│   │   ├── chat.py      # 聊天查询路由
│   │   └── auth.py      # 认证路由
│   └── middleware/      # 中间件
│       ├── __init__.py
│       └── auth.py
├── ui/                  # 前端界面 - React应用
│   ├── public/
│   ├── src/
│   │   ├── components/  # React组件
│   │   ├── pages/       # 页面组件
│   │   ├── services/    # API服务
│   │   └── utils/       # 工具函数
│   ├── package.json
│   └── README.md
├── core/                # 核心业务逻辑
│   ├── __init__.py
│   ├── document_processor.py # 文档处理核心
│   ├── embedding_engine.py   # 嵌入引擎
│   ├── retrieval_engine.py   # 检索引擎
│   └── generation_engine.py  # 生成引擎
├── utils/               # 工具函数
│   ├── __init__.py
│   ├── database.py      # 数据库工具
│   ├── security.py      # 安全工具
│   ├── logger.py        # 日志工具
│   └── validators.py    # 数据验证器
├── tests/               # 测试文件
│   ├── __init__.py
│   ├── test_core.py     # 核心功能测试
│   ├── test_api.py      # API测试
│   └── conftest.py      # 测试配置
├── docs/                # 文档
│   ├── api.md           # API文档
│   ├── deployment.md    # 部署指南
│   └── user_guide.md    # 用户指南
├── config/              # 配置文件
│   ├── __init__.py
│   ├── database.py      # 数据库配置
│   ├── settings.py      # 应用配置
│   └── constants.py     # 常量定义
├── uploads/             # 上传文件存储目录
├── vector_store/        # 向量数据库存储目录
├── requirements.txt     # Python依赖
├── pyproject.toml       # 项目配置
├── .env                 # 环境变量
├── docker-compose.yml   # Docker编排文件
└── README.md            # 项目说明