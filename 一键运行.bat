@echo off
chcp 65001 > nul
echo ========================================================
echo        动漫评分预测系统 - 一键运行脚本
echo ========================================================

echo [1] 检查并安装依赖项...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 安装依赖失败，请检查 Python 及 Pip 环境是否配置正确。
    pause
    exit /b %errorlevel%
)

echo.
echo [2] 正在生成数据可视化图表 (outputs/figures/)...
python scripts\visualize.py

echo.
echo [3] 正在执行受众累积分布分析...
python scripts\analyze.py

echo.
echo [4] 开始完整流水线 (加载数据-构图-训练GNN-基线对比-消融实验)...
python scripts\train.py

echo.
echo ========================================================
echo    所有任务执行完毕！模型权重、实验 CSV 及图表均存储在 outputs/
echo ========================================================
pause
