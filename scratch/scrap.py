
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


    def transform(self, X):
        """Transform X to a previously fit embedded space.
        A previous training set must have already been fit.
        The new gradient is calculated using contributions from
        previously fit data, but only the new data is transformed.
        This is not the equivalent of running fit_transform,
        since calling fit(X) followed by transform(X) runs the
        gradient calculation twice, once for just X and the second time
        on the concatenated array [X, X].
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self._check_fitted()
        if np.allclose(X, self.training_data_, rtol=1e-4):
            warnings.warn("The transform input appears to be similar "
                          "to previously fit data. This can result in "
                          "duplicated data; consider using fit_transform")

        skip_num_points = self.embedding_.shape[0]
        full_set = np.vstack((self.embedding_, X))
        Xt = self._fit(full_set, skip_num_points=skip_num_points)
        return Xt
