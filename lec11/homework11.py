import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.
    '''
    r = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = r.record(source)

    text = r.recognize_google(audio, language=language)
    return text
