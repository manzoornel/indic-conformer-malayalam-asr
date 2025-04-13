
import gradio as gr
from IndicASR import load_model, predict

# Load Malayalam ASR model
model = load_model(lang="ml")

def transcribe(audio_path):
    try:
        return predict(model, audio_path)
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="upload", type="filepath", label="Upload Malayalam Audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="Malayalam Speech-to-Text",
    description="Upload a Malayalam audio file. AI4Bharat Indic Conformer ASR will transcribe it."
)

iface.launch()
