import gradio as gr
print(f"🔴 当前加载的 Gradio 版本: {gr.__version__}") # <--- 加上这一句！
if int(gr.__version__.split('.')[0]) < 4:
    raise ImportError("❌ Gradio 版本过低！请重新安装。")