        lstmf_layer = bilstm_layer.forward_layer
        lstmf_layer.return_sequences = True
        lstmf_model = Model(inputs=model.input, outputs=lstmf_layer.output)
        lstmf_test = lstmf_model.predict( [X_test[:,:,7:], X_test[:,:,:7]] )
#        lstmf_test = lstmf_model.predict( X_test )
        print(len(lstmf_test))
        print(lstmf_test.shape)
        raise

        lstmb_layer = bilstm_layer.backward_layer
        lstmb_layer.return_state = True
        lstmb_layer.return_sequences = True
#        bilstm_new = Bidirectional(lstmf_layer, backward_layer=lstmb_layer)(\
#                dropout_layer)
#        lstm_model = Model(inputs=model.input, outputs=bilstm_new)
#        lstm_test = lstm_model.predict( [X_test[:,:,7:], X_test[:,:,:7]] )
#        raise

        print( "With return state/sequences, forward LSTM:" )
        seq_h,last_fh,last_fc = lstmf_layer( [X_test[:,:,7:], X_test[:,:,:7]] )
        print(f"\tseq_h: {seq_h.shape}")
        print(f"\tlast_fh: {last_fh.shape}")
        print(f"\tlast_fc: {last_fc.shape}")
        print( "With return state/sequences, backward LSTM:" )
        seq_h,last_bh,last_bc = lstmb_layer( [X_test[:,:,7:], X_test[:,:,:7]] )
        print(f"\tseq_h: {seq_h.shape}")
        print(f"\tlast_bh: {last_bh.shape}")
        print(f"\tlast_bc: {last_bc.shape}")
        raise

        bilstm_layer.return_state = True
        bilstm_layer.return_sequences = True
        bilstm_layer.forward_layer.return_state = True
        bilstm_layer.forward_layer.return_sequences = True
        bilstm_layer.backward_layer.return_state = True
        bilstm_layer.backward_layer.return_sequences = True

        seq_h,last_fh,last_fc,last_bh,last_bc = bilstm_layer( \
                [X_test[:,:,7:], X_test[:,:,:7]] )
        print(f"\tseq_h: {seq_h.shape}")
        print(f"\tlast_fh: {last_fh.shape}")
        print(f"\tlast_fc: {last_fc.shape}")
        print(f"\tlast_bh: {last_bh.shape}")
        print(f"\tlast_bc: {last_bc.shape}")
        raise

