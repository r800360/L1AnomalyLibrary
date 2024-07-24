import numpy as np
import tensorflow as tf

import L1AnomalyBase

#Utility class for large encoder methods

class L1AnomalyEncoderUtil:
    
    def setup_constants(self):
        #Constants
        self.nmet = 1
        self.nele = 4
        self.nmu = 4
        self.njet = 10
        self.ele_off = 1
        self.mu_off = self.nmet + self.nele
        self.jet_off = self.nmet + self.nele + self.nmu
        self.phi_max = np.pi
        self.ele_eta_max = 3.0
        self.mu_eta_max = 2.1
        self.jet_eta_max = 4.0
    
    def mse_loss(self, inputs, outputs):
        return tf.math.reduce_mean((inputs - outputs) ** 2, axis=-1)
    
    def get_loss(self, model, X, X_scaled, variational = False):
        if variational == False:
            return np.array(self.make_mse_per_sample(X_scaled, model.predict(X, batch_size=1024)))
        else:
            return np.array(self.mod_make_mse_per_sample(X_scaled, model.predict(X, batch_size=1024)))
    
    def make_mse(self, inputs, outputs):
        loss = self.make_mse_per_sample(inputs, outputs)
        loss = tf.math.reduce_mean(loss, axis=0)
        loss = tf.cast(loss, dtype=inputs.dtype)
        return loss
    
    def make_mse_per_sample_ae(self, inputs, outputs):
        outputs = tf.cast(outputs, dtype=inputs.dtype)
        inputs = tf.reshape(inputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])
        outputs = tf.reshape(outputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])
        outputs_pt = outputs[:, :, 0]
        outputs_phi = self.phi_max * tf.math.tanh(outputs[:, :, 2])
        outputs_eta_met = outputs[:, 0:1, 1]
        outputs_eta_ele = self.ele_eta_max * tf.math.tanh(outputs[:, self.ele_off : self.ele_off + self.nele, 1])
        outputs_eta_mu = self.mu_eta_max * tf.math.tanh(outputs[:, self.mu_off : self.mu_off + self.nmu, 1])
        outputs_eta_jet = self.jet_eta_max * tf.math.tanh(outputs[:, self.jet_off : self.jet_off + self.njet, 1])
        outputs_eta = tf.concat([outputs_eta_met, outputs_eta_ele, outputs_eta_mu, outputs_eta_jet], axis=1)
        outputs = tf.stack([outputs_pt, outputs_eta, outputs_phi], axis=-1)
        mask = tf.math.not_equal(inputs, 0)
        mask = tf.cast(mask, dtype=outputs.dtype)
        outputs = mask * outputs
        loss = self.mse_loss(tf.reshape(inputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
                            tf.reshape(outputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]))
        return loss

    def make_mse_per_sample_ae_class(self, inputs, outputs):
        outputs = tf.cast(outputs, dtype=inputs.dtype)  # make inputs and outputs same type

        #1+4+4+10 = 19 with 3 features of pT, eta, phi which are transverse momentum, pseduorapidity, azimuthal angle
        # as in Main AE paper
        inputs = tf.reshape(inputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])
        outputs = tf.reshape(outputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])

        # extract pt
        outputs_pt = outputs[:, :, 0]
        
        # extract class
        outputs_class = outputs[:, :, 3]

        # trick with phi (rescaled tanh activation function) - pi times tanh of azimuthal angle
        outputs_phi = self.phi_max * tf.math.tanh(outputs[:, :, 2])

        #Extracts missing transverse energy pseudorapidity outputs
        outputs_eta_met = outputs[:, 0:1, 1]

        # trick with eta (rescaled tanh activation function) - max electron pseudorapidity times tanh of pseudorapidity
        outputs_eta_ele = self.ele_eta_max * tf.math.tanh(
            outputs[:, self.ele_off : self.ele_off + self.nele, 1]
        )

        #Treatment of muon pseudorapidity analogous to that of electron
        outputs_eta_mu = self.mu_eta_max * tf.math.tanh(outputs[:, self.mu_off : self.mu_off + self.nmu, 1])

        #Treatment of jet pseudorapidity analogous to that of electrons and muons
        outputs_eta_jet = self.jet_eta_max * tf.math.tanh(
            outputs[:, self.jet_off : self.jet_off + self.njet, 1]
        )

        #Output psuedorapidity is triple with missing transverse eneergy, electron, muon, jet
        outputs_eta = tf.concat(
            [outputs_eta_met, outputs_eta_ele, outputs_eta_mu, outputs_eta_jet], axis=1
        )

        # use both tricks - stacks into standard triple - transvere momenta, pseudorapidity, azimuthal angle
        outputs = tf.stack([outputs_pt, outputs_eta, outputs_phi, outputs_class], axis=-1)

        # mask zero features - Zero Padding after output formation
        mask = tf.math.not_equal(inputs, 0)
        mask = tf.cast(mask, dtype=outputs.dtype)
        outputs = mask * outputs

        #Apply previously defined MSE_loss function 1 - corresponding to nmet
        loss = self.mse_loss(
            tf.reshape(inputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
            tf.reshape(outputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
        )
        
        return loss
        
    def make_mse_per_sample_vae(self, inputs, outputs):
        mainOutputs = tf.cast(outputs[:, 3:-3], dtype=inputs.dtype)  # make inputs and outputs same type
        meanLatentSpaceVector = outputs[:, :3]
        logVarVector = outputs[:, -3:]
        beta = 0.5
        klDivCoef = (beta) * -0.5

        #1+4+4+10 = 19 with 3 features of pT, eta, phi which are transverse momentum, pseduorapidity, azimuthal angle
        # as in Main AE paper
        inputs = tf.reshape(inputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])
        mainOutputs = tf.reshape(mainOutputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])

        # extract pt
        outputs_pt = mainOutputs[:, :, 0]

        # trick with phi (rescaled tanh activation function) - pi times tanh of azimuthal angle
        outputs_phi = self.phi_max * tf.math.tanh(mainOutputs[:, :, 2])

        #Extracts missing transverse energy pseudorapidity outputs
        outputs_eta_met = mainOutputs[:, 0:1, 1]

        # trick with eta (rescaled tanh activation function) - max electron pseudorapidity times tanh of pseudorapidity
        outputs_eta_ele = self.ele_eta_max * tf.math.tanh(
            mainOutputs[:, self.ele_off : self.ele_off + self.nele, 1]
        )

        #Treatment of muon pseudorapidity analogous to that of electron
        outputs_eta_mu = self.mu_eta_max * tf.math.tanh(mainOutputs[:, self.mu_off : self.mu_off + self.nmu, 1])

        #Treatment of jet pseudorapidity analogous to that of electrons and muons
        outputs_eta_jet = self.jet_eta_max * tf.math.tanh(
            mainOutputs[:, self.jet_off : self.jet_off + self.njet, 1]
        )

        #Output psuedorapidity is triple with missing transverse eneergy, electron, muon, jet
        outputs_eta = tf.concat(
            [outputs_eta_met, outputs_eta_ele, outputs_eta_mu, outputs_eta_jet], axis=1
        )

        # use both tricks - stacks into standard triple - transvere momenta, pseudorapidity, azimuthal angle
        mainOutputs = tf.stack([outputs_pt, outputs_eta, outputs_phi], axis=-1)

        # mask zero features - Zero Padding after output formation
        mask = tf.math.not_equal(inputs, 0)
        mask = tf.cast(mask, dtype=mainOutputs.dtype)
        mainOutputs = mask * mainOutputs

        #Apply previously defined MSE_loss function 1 - corresponding to nmet
        mse_loss_value = self.mse_loss(
            tf.reshape(inputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
            tf.reshape(mainOutputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
        )
        mse_loss_value = tf.math.reduce_mean(mse_loss_value, axis=0)

        kl_divergence = tf.math.multiply(klDivCoef, tf.reduce_sum(1 + logVarVector - tf.square(meanLatentSpaceVector) - tf.exp(logVarVector), axis=-1))
        kl_divergence = tf.math.reduce_mean(kl_divergence, axis=0)
        kl_divergence = tf.cast(kl_divergence, dtype = mse_loss_value.dtype)

        # Apply previously defined MSE_loss function 1 - corresponding to nmet
        loss = tf.math.add(tf.math.multiply((1 - beta), mse_loss_value), kl_divergence)
        return loss
        
    def make_mse_per_sample_vae_class(self, inputs, outputs):
        mainOutputs = tf.cast(outputs[:, 2:-2], dtype=inputs.dtype)  # make inputs and outputs same type
        meanLatentSpaceVector = outputs[:, :2]
        logVarVector = outputs[:, -2:]
        beta = 0.5
        klDivCoef = (beta) * -0.5

        #1+4+4+10 = 19 with 3 features of pT, eta, phi which are transverse momentum, pseduorapidity, azimuthal angle
        # as in Main AE paper
        inputs = tf.reshape(inputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])
        mainOutputs = tf.reshape(mainOutputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])

        # extract pt
        outputs_pt = mainOutputs[:, :, 0]
        
        # extract class
        outputs_class = mainOutputs[:, :, 3]
        
        # trick with phi (rescaled tanh activation function) - pi times tanh of azimuthal angle
        outputs_phi = self.phi_max * tf.math.tanh(mainOutputs[:, :, 2])

        #Extracts missing transverse energy pseudorapidity outputs
        outputs_eta_met = mainOutputs[:, 0:1, 1]

        # trick with eta (rescaled tanh activation function) - max electron pseudorapidity times tanh of pseudorapidity
        outputs_eta_ele = self.ele_eta_max * tf.math.tanh(
            mainOutputs[:, self.ele_off : self.ele_off + self.nele, 1]
        )

        #Treatment of muon pseudorapidity analogous to that of electron
        outputs_eta_mu = self.mu_eta_max * tf.math.tanh(mainOutputs[:, self.mu_off : self.mu_off + self.nmu, 1])

        #Treatment of jet pseudorapidity analogous to that of electrons and muons
        outputs_eta_jet = self.jet_eta_max * tf.math.tanh(
            mainOutputs[:, self.jet_off : self.jet_off + self.njet, 1]
        )

        #Output psuedorapidity is triple with missing transverse eneergy, electron, muon, jet
        outputs_eta = tf.concat(
            [outputs_eta_met, outputs_eta_ele, outputs_eta_mu, outputs_eta_jet], axis=1
        )

        # use both tricks - stacks into standard triple - transvere momenta, pseudorapidity, azimuthal angle
        mainOutputs = tf.stack([outputs_pt, outputs_eta, outputs_phi, outputs_class], axis=-1)

        # mask zero features - Zero Padding after output formation
        mask = tf.math.not_equal(inputs, 0)
        mask = tf.cast(mask, dtype=mainOutputs.dtype)
        mainOutputs = mask * mainOutputs

        #Apply previously defined MSE_loss function 1 - corresponding to nmet
        mse_loss_value = self.mse_loss(
            tf.reshape(inputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
            tf.reshape(mainOutputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
        )
        mse_loss_value = tf.math.reduce_mean(mse_loss_value, axis=0)

        kl_divergence = tf.math.multiply(klDivCoef, tf.reduce_sum(1 + logVarVector - tf.square(meanLatentSpaceVector) - tf.exp(logVarVector), axis=-1))
        kl_divergence = tf.math.reduce_mean(kl_divergence, axis=0)
        kl_divergence = tf.cast(kl_divergence, dtype = mse_loss_value.dtype)

        # Apply previously defined MSE_loss function 1 - corresponding to nmet
        loss = tf.math.add(tf.math.multiply((1 - beta), mse_loss_value), kl_divergence)
        
        return loss
        

    def make_mse_per_sample(self, inputs, outputs, variational = False, classVar = False):
        self.setup_constants()
        
        if variational == False:
            if classVar == False:
                return self.make_mse_per_sample_ae(self, inputs, outputs)
            else:
                return self.make_mse_per_sample_ae_class(self, inputs, outputs)
        else:
            if classVar == False:
                return self.make_mse_per_sample_vae(self, inputs, outputs)
            else:
                return self.make_mse_per_sample_vae_class(self, inputs, outputs)

    
    #For use purely for DNNVAE
    def mod_make_mse_per_sample(self, inputs, outputs, classVar = False):
        if classVar == False:
            mainOutputs = tf.cast(outputs[:, 3:-3], dtype=inputs.dtype)  # make inputs and outputs same type
            meanLatentSpaceVector = outputs[:, :3]
            logVarVector = outputs[:, -3:]
            beta = 0.5
            klDivCoef = (beta) * -0.5

            #1+4+4+10 = 19 with 3 features of pT, eta, phi which are transverse momentum, pseduorapidity, azimuthal angle
            # as in Main AE paper
            inputs = tf.reshape(inputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])
            mainOutputs = tf.reshape(mainOutputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])

            # extract pt
            outputs_pt = mainOutputs[:, :, 0]

            # trick with phi (rescaled tanh activation function) - pi times tanh of azimuthal angle
            outputs_phi = self.phi_max * tf.math.tanh(mainOutputs[:, :, 2])

            #Extracts missing transverse energy pseudorapidity outputs
            outputs_eta_met = mainOutputs[:, 0:1, 1]

            # trick with eta (rescaled tanh activation function) - max electron pseudorapidity times tanh of pseudorapidity
            outputs_eta_ele = self.ele_eta_max * tf.math.tanh(
                mainOutputs[:, self.ele_off : self.ele_off + self.nele, 1]
            )

            #Treatment of muon pseudorapidity analogous to that of electron
            outputs_eta_mu = self.mu_eta_max * tf.math.tanh(mainOutputs[:, self.mu_off : self.mu_off + self.nmu, 1])

            #Treatment of jet pseudorapidity analogous to that of electrons and muons
            outputs_eta_jet = self.jet_eta_max * tf.math.tanh(
                mainOutputs[:, self.jet_off : self.jet_off + self.njet, 1]
            )

            #Output psuedorapidity is triple with missing transverse eneergy, electron, muon, jet
            outputs_eta = tf.concat(
                [outputs_eta_met, outputs_eta_ele, outputs_eta_mu, outputs_eta_jet], axis=1
            )

            # use both tricks - stacks into standard triple - transvere momenta, pseudorapidity, azimuthal angle
            mainOutputs = tf.stack([outputs_pt, outputs_eta, outputs_phi], axis=-1)

            # mask zero features - Zero Padding after output formation
            mask = tf.math.not_equal(inputs, 0)
            mask = tf.cast(mask, dtype=mainOutputs.dtype)
            mainOutputs = mask * mainOutputs

            #Apply previously defined MSE_loss function 1 - corresponding to nmet
            mse_loss_value = self.mse_loss(
                tf.reshape(inputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
                tf.reshape(mainOutputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
            )
            #mse_loss_value = tf.math.reduce_mean(mse_loss_value, axis=0)

            kl_divergence = tf.math.multiply(klDivCoef, tf.reduce_sum(1 + logVarVector - tf.square(meanLatentSpaceVector) - tf.exp(logVarVector), axis=-1))
            #kl_divergence = tf.math.reduce_mean(kl_divergence, axis=0)
            kl_divergence = tf.cast(kl_divergence, dtype = mse_loss_value.dtype)

            # Apply previously defined MSE_loss function 1 - corresponding to nmet
            loss = tf.math.add(tf.math.multiply((1 - beta), mse_loss_value), kl_divergence)
            return loss
        else:
            #Need to find the right code here
            raise NotImplementedError
        
            mainOutputs = tf.cast(outputs[:, 2:-2], dtype=inputs.dtype)  # make inputs and outputs same type
            meanLatentSpaceVector = outputs[:, :2]
            logVarVector = outputs[:, -2:]
            beta = 0.5
            klDivCoef = (beta) * -0.5

            #1+4+4+10 = 19 with 3 features of pT, eta, phi which are transverse momentum, pseduorapidity, azimuthal angle
            # as in Main AE paper
            inputs = tf.reshape(inputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])
            mainOutputs = tf.reshape(mainOutputs, [-1, (self.nmet + self.nele + self.nmu + self.njet), self.nfeat])

            # extract pt
            outputs_pt = mainOutputs[:, :, 0]

            # trick with phi (rescaled tanh activation function) - pi times tanh of azimuthal angle
            outputs_phi = self.phi_max * tf.math.tanh(mainOutputs[:, :, 2])

            #Extracts missing transverse energy pseudorapidity outputs
            outputs_eta_met = mainOutputs[:, 0:1, 1]

            # trick with eta (rescaled tanh activation function) - max electron pseudorapidity times tanh of pseudorapidity
            outputs_eta_ele = self.ele_eta_max * tf.math.tanh(
                mainOutputs[:, self.ele_off : self.ele_off + self.nele, 1]
            )

            #Treatment of muon pseudorapidity analogous to that of electron
            outputs_eta_mu = self.mu_eta_max * tf.math.tanh(mainOutputs[:, self.mu_off : self.mu_off + self.nmu, 1])

            #Treatment of jet pseudorapidity analogous to that of electrons and muons
            outputs_eta_jet = self.jet_eta_max * tf.math.tanh(
                mainOutputs[:, self.jet_off : self.jet_off + self.njet, 1]
            )

            #Output psuedorapidity is triple with missing transverse eneergy, electron, muon, jet
            outputs_eta = tf.concat(
                [outputs_eta_met, outputs_eta_ele, outputs_eta_mu, outputs_eta_jet], axis=1
            )

            # use both tricks - stacks into standard triple - transvere momenta, pseudorapidity, azimuthal angle
            mainOutputs = tf.stack([outputs_pt, outputs_eta, outputs_phi], axis=-1)

            # mask zero features - Zero Padding after output formation
            mask = tf.math.not_equal(inputs, 0)
            mask = tf.cast(mask, dtype=mainOutputs.dtype)
            mainOutputs = mask * mainOutputs

            #Apply previously defined MSE_loss function 1 - corresponding to nmet
            mse_loss_value = self.mse_loss(
                tf.reshape(inputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
                tf.reshape(mainOutputs, [-1, (1 + self.nele + self.nmu + self.njet) * self.nfeat]),
            )
            #mse_loss_value = tf.math.reduce_mean(mse_loss_value, axis=0)

            kl_divergence = tf.math.multiply(klDivCoef, tf.reduce_sum(1 + logVarVector - tf.square(meanLatentSpaceVector) - tf.exp(logVarVector), axis=-1))
            #kl_divergence = tf.math.reduce_mean(kl_divergence, axis=0)
            kl_divergence = tf.cast(kl_divergence, dtype = mse_loss_value.dtype)

            # Apply previously defined MSE_loss function 1 - corresponding to nmet
            loss = tf.math.add(tf.math.multiply((1 - beta), mse_loss_value), kl_divergence)
            return loss