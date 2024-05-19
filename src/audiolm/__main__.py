# """Entry point for the app.

# This can be invoked either by using

# python -m audiolm

# or via as a cli application (TODO) by using

# audiolm-cli
# """

# from audiolm.data_preparation import AudioDataLoader
# from audiolm.w2v_hubert import W2VHuBert

# if __name__ == "__main__":

#     ##Semantic stage##
#     dataloader = AudioDataLoader(
#         data_path="/Users/valerio/Desktop/ei",
#         batch_size=5,
#         shuffle=False,
#         max_length_audio=3,
#     )
#     for batch in dataloader:
#         print(batch.shape)
#     print(dataloader.__len__())

#     hubert = W2VHuBert(
#         dataloader=dataloader, sample_frequency=16000, input_audio_hz=16000
#     )
#     batch_1 = next(iter(dataloader))
#     semantic_tokens = hubert.forward(batch_1)
#     print(semantic_tokens.shape)

#     # for token in semantic_tokens:
#     #     inputs =
