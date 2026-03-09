import gradio as gr
from .g2p_phonemizer import g2p_phonemizer_en

iface = gr.Interface(
    fn=g2p_phonemizer_en,
    inputs=gr.Textbox(lines=2, placeholder="Enter English text...", label="Input Text"),
    outputs=gr.Textbox(label="IPA Phonemes"),
    examples=[["Hello world"], ["How are you?"], ["Test sentence."]],
    live=True,  # Enables real-time updates
    title="English Text to IPA Converter",
    description="Convert English text to IPA phonemes using grapheme-to-phoneme conversion"
)

if __name__ == "__main__":
    iface.launch()