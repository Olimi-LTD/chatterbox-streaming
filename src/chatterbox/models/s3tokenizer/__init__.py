from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1


def drop_invalid_tokens(x):
    """Drop SoS and EoS - Fixed version"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"

    # Handle the SOS token
    if SOS in x:
        sos_indices = (x == SOS).nonzero(as_tuple=True)[0]
        if len(sos_indices) > 0:
            s = sos_indices[0].item() + 1  # Convert to Python int using .item()
        else:
            s = 0
    else:
        s = 0

    # Handle the EOS token
    if EOS in x:
        eos_indices = (x == EOS).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            e = eos_indices[0].item()  # Convert to Python int using .item()
        else:
            e = None
    else:
        e = None

    # Now s and e are Python integers, so slicing will work
    x = x[s:e]
    return x

#def drop_invalid_tokens(x):
#    """Drop SoS and EoS"""
#    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
#    if SOS in x:
#        s = (x == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
#    else:
#        s = 0
#
#    if EOS in x:
#        e = (x == EOS).nonzero(as_tuple=True)[0].squeeze(0)
#    else:
#        e = None
#
#    x = x[s: e]
#    return x

