from manim import *

class MathScene(Scene):
    def construct(self):
        # Biểu thức cần vẽ
        expression = MathTex("E = mc^2")  # Thay đổi biểu thức tại đây
        self.play(Write(expression))
        self.wait(2)

if __name__ == "__main__":
    from manim import config
    config.media_width = "75%"
    MathScene().render()
