from pocket_tts import TTSModel
import scipy.io.wavfile

"""

Dialogues

EN
Calibration phase number 1: Look straight ahead with your eyes comfortably and normally open; please avoid blinking
Calibration phase number 2: Look straight ahead with your eyes comfortably closed; please avoid blinking

ES
Primera fase de calibración: Mira fijamente hacia enfrente, sin parpadear, con los ojos abiertos cómodamente
Segunda fase de calibración: Mira fijamente hacia enfrente, sin parpadear, con los ojos cerrados cómodamente
"""

# Model loading
tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt("caro_davy")

# Audio strings -- Populate as needed
audios = {
    "calib_phase_1": "Calibration phase number 1: Look straight ahead with your eyes comfortably and normally open; please avoid blinking",
    "calib_phase_2": "Calibration phase number 2: Look straight ahead with your eyes comfortably closed; please avoid blinking"
}


for audio in audios:
    audio_data = tts_model.generate_audio(voice_state, audios[audio])
    scipy.io.wavfile.write(f"morphed_system4/distraction/instruction_generation/output/{audio}.wav", tts_model.sample_rate, audio_data.numpy())
