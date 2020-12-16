import numpy as np

class KF:
    F = np.array([
        [1., 0, 0, 0, 1, 0, 0],
        [0., 1, 0, 0, 0, 1, 0],
        [0., 0, 1, 0, 0, 0, 1],
        [0., 0, 0, 1, 0, 0, 0],
        [0., 0, 0, 0, 1, 0, 0],
        [0., 0, 0, 0, 0, 1, 0],
        [0., 0, 0, 0, 0, 0, 1]
    ])
    H = np.eye(7, dtype=float)
    Q = np.eye(7, dtype=float)
    R = np.diag([5., 5, 25, 5, 5, 5, 5])

    def __init__(self, init_states):
        self.states = init_states # dict from `tracker.py` label to np.array (vec)
        self.live_states = set(init_states.keys())
        self.Pp = {
            l: np.diag([2., 2., 4., 1., 1000, 1000, 1000]) for l in init_states.keys()
         }           # dict from `tracker.py` label to np.array (mat)


    def predict(self):
        predicted_states = {}
        for label in self.live_states:
            predicted_states[label] = np.dot(KF.F, self.states[label])
        return predicted_states


    def latest_live_states(self):
        states = {}
        for label in self.live_states:
            states[label] = self.states[label]
        return states

    
    def birth_state(self, label, state):
        self.live_states.add(label)
        self.states[label] = state
        self.Pp[label] = np.diag([2., 2., 4., 1., 1000, 1000, 1000])


    def kill_state(self, label):
        self.live_states.remove(label)


    def update(self, ys, preds):
        for label in preds:
            if not label in ys:
                self.states[label] = preds[label]
                self.Pp[label] = KF.F@self.Pp[label]@np.transpose(KF.F)+KF.Q
            else:
                y = ys[label]
                pred = preds[label]
                Pp = self.Pp[label]
                Pkm = KF.F@Pp@np.transpose(KF.F)+KF.Q
                K = Pkm@np.transpose(KF.H)@np.linalg.inv(KF.H@Pkm@np.transpose(KF.H)+KF.R)

                self.states[label] = pred+K.dot(y-KF.H.dot(pred))

                t1 = K@KF.H
                t2 = np.eye(t1.shape[0])
                t3 = t2-t1
                self.Pp[label] = t3@Pkm@np.transpose(t3)+K@KF.R@np.transpose(K)
