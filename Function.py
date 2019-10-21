import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


def Generating_Kernel(Feature_Matrix, Kernel_type, power=10, BW=10):
    num_samples = Feature_Matrix.shape[0]
    if Kernel_type == "Gaussian_Kernel":
        Gaussian_Kernel = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                # Gaussian_Kernel[i, j] = np.exp((-1 * np.sum((Feature_Matrix[i] - Feature_Matrix[j]) ** 2)) / (2 * BW ** 2))
                Gaussian_Kernel[i, j] = np.exp(
                    (-1 * LA.norm(Feature_Matrix[i] - Feature_Matrix[j]) ** 2) / (2 * BW ** 2))
        return Gaussian_Kernel

    elif Kernel_type == "Quick_Gaussian_Kernel":
        square = np.sum(Feature_Matrix ** 2, axis=1)
        column_vec = square[:, np.newaxis]
        row_vec = square[np.newaxis, :]
        Gaussian_Kernel = np.exp(
            -1 * (-2 * Feature_Matrix.dot(Feature_Matrix.T) + column_vec + row_vec) / (2 * BW ** 2))

        return Gaussian_Kernel
    elif Kernel_type == "Linear_Kernel":
        return Feature_Matrix.dot(Feature_Matrix.T)
    elif Kernel_type == "Polynomial_Kernel":
        return (Feature_Matrix.dot(Feature_Matrix.T) + 1) ** power


def Generating_Synthetic_Set(num_samples, num_features, Epsilons, bias=False):
    Epsilon_1, Epsilon_2 = Epsilons
    X = np.zeros((num_samples, num_features))
    for i in range(num_samples):
        while (True):
            X[i] = np.random.multivariate_normal(np.zeros(num_features), np.eye(num_features))
            _lambda = Epsilon_1 * np.sum(X[i, :] ** 2) + Epsilon_2 * np.sum(X[i, :])
            if _lambda > 0:
                break

    survival_times = np.zeros(num_samples)  # the survival time of each employees
    for i in range(num_samples):
        age = np.random.exponential(Epsilon_1 * np.sum(X[i, :] ** 2) + Epsilon_2 * np.sum(X[i, :]), size=1)
        survival_times[i] = np.ceil(age)

    if bias == True:
        bias_coefficients = np.ones(num_samples)  # [:, np.newaxis]
        X = np.c_[X, bias_coefficients]

    num_tasks = int(max(survival_times))

    Y = np.ones((num_samples, num_tasks))
    # the lifetime matrix of all employees, if one employee leave at the time interval k, then from Y[i,k](inlcude)  all entries are -1
    for i in range(num_samples):
        Y[i, int(survival_times[i]):] = -1
    return X, Y, survival_times


def Computing_loss_Single_Task(W, X, Y):
    W_norm = LA.norm(W)
    loss = (W_norm ** 2) / 2
    Y_pred = X.dot(W)
    violated_mask = Y_pred * Y < 1
    l1_loss = np.sum(1 - (Y * Y_pred)[violated_mask])
    loss += l1_loss

    return loss


def Validate_Synthetic_DataSet(Kernel_Matrix, survival_times):
    age_gap_list = []
    for i in range(10):
        [row_indices, col_indices] = np.where((Kernel_Matrix >= i * 0.1) & (Kernel_Matrix <= (i + 1) * 0.1))
        num_pairs = row_indices.shape[0]
        age_gap_list.append(abs(survival_times[row_indices] - survival_times[col_indices]))
        print("There are ", num_pairs,
              "pairs of samples of similarity between {:.1f} and {:.1f}".format(0.1 * i, 0.1 * (i + 1)),
              "which have an average age_gap ", np.mean(age_gap_list[-1]),
              "and the median is:", np.median(age_gap_list[-1]))
    plt.boxplot(age_gap_list, patch_artist=True)
    plt.ylim(0, 200)
    plt.show()


def Ages(Y):
    num_samples = Y.shape[0]
    num_tasks = Y.shape[1]
    predict_age = []
    for i in range(num_samples):
        states = Y[i]
        [violated_indices] = np.where(states < 0)
        age = num_tasks if len(violated_indices) == 0 else violated_indices[0]
        predict_age.append(age)
    return predict_age


def Calculate_C_index(Age_gt, Age_pred):
    num_samples = len(Age_gt)
    useful_pairs = 0
    denominator = num_samples * (num_samples - 1) / 2
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if Age_gt[i] == Age_gt[j]:
                denominator -= 1
            elif (Age_gt[i] - Age_gt[j]) * (Age_pred[i] - Age_pred[j]) > 0:
                useful_pairs += 1
    print(useful_pairs, denominator)
    c_index = useful_pairs / denominator
    return c_index


def Non_Kernel_Pegasos(batch_size, X, W, Y, loss_list, iter_times, weight_decay):
    eta = 1 / (iter_times * weight_decay)
    num_samples = X.shape[0]
    num_features = X.shape[1]
    num_tasks = W.shape[1]

    dL1_W = np.zeros((num_features, num_tasks))

    # select a mini batch
    IDs = np.random.rand(batch_size, 1) * num_samples
    IDs = IDs.astype(int).reshape(-1)
    x = X[IDs]
    y = Y[IDs]

    # find the violated entries
    y_hat = x.dot(W)
    [row_indices, col_indices] = np.where(y_hat * y < 1)
    num_violates = len(row_indices)

    # Compute dL1_w
    for i in range(num_violates):
        dW = np.zeros((num_features, num_tasks))
        dW[:, col_indices[i]] = x[row_indices[i]] * y[row_indices[i], col_indices[i]]
        dL1_W += dW

    temp_W = W - 1 / iter_times * W + eta / batch_size * dL1_W
    projection_coefficient = np.min([1 / ((weight_decay ** 0.5) * LA.norm(temp_W)), 1])
    temp_W *= projection_coefficient
    # now make prediction
    Y_pred = np.sign(X.dot(temp_W))

    # compute the loss
    W_norm = LA.norm(W)
    predict_result = Y_pred * Y
    mask = predict_result < 1
    l1_loss = np.mean(1 - predict_result[mask])
    # l1_loss = np.sum(np.sum(Y_pred * Y < 1))
    loss = W_norm * W_norm / 2 + l1_loss

    # compare with previous loss
    # if iter_times == 1 or loss < loss_list[-1] or num_violates < num_violates_list[-1]:
    loss_list.append(loss)
    W = temp_W

    return W, loss_list


def Non_Kernel_2c():
    pass


def Split_Non_Kernel_Pegasos(batch_size, X, W, Y, loss_list, iter_times, weight_decay):
    eta = 1 / (iter_times * weight_decay)
    num_samples = X.shape[0]
    num_tasks = W.shape[1]

    # select a mini batch
    IDs = np.random.rand(batch_size, 1) * num_samples
    IDs = IDs.astype(int).reshape(-1)
    x = X[IDs]
    y = Y[IDs]

    # find the violated entries
    y_hat = x.dot(W)
    [row_indices, col_indices] = np.where(y_hat * y < 1)

    indices = np.array(np.c_[row_indices, col_indices])

    indices = indices[indices[:, 1].argsort()]
    # print(indices)
    W -= eta * weight_decay * W

    for i in range(num_tasks):
        mask = indices[:, 1] == i
        if np.sum(mask == True) == 0:
            continue

        id = indices[mask][:, 0]
        temp_W = W
        temp_W[:, i] += np.sum(x[id] * y[id][:, i][:, None], axis=0)
        beforeLoss = Computing_loss_Single_Task(W[:, i], X, Y[:, i])
        afterLoss = Computing_loss_Single_Task(temp_W[:, i], X, Y[:, i])
        if afterLoss < beforeLoss:
            loss_list[i].append()
            W[:, i] += np.sum(x[id] * y[id], axis=0)

    projection_coefficient = np.min([1 / ((weight_decay ** 0.5) * LA.norm(W)), 1])
    W *= projection_coefficient

    return W


def Kernel_Pegasos_nonchecking(batch_size, Kernel_Matrix, alpha, Y, iter_times, weight_decay):
    ###################
    # Define parameters
    ###################
    num_samples = Kernel_Matrix.shape[0]
    num_tasks = Y.shape[1]

    #######################
    # Define the mini-batch
    #######################
    IDs = np.random.rand(batch_size, 1) * num_samples
    IDs = IDs.astype(int).reshape(-1)

    ###########################################
    # Make prediction
    ###########################################

    haty_IDs = (Kernel_Matrix[IDs].dot(alpha * Y))
    haty_IDs /= (iter_times * weight_decay)

    # update alpha
    mask = Y[IDs] * haty_IDs < 1
    alpha[IDs] += mask

    return alpha


def Kernel_Pegasos(batch_size, Kernel_Matrix, alpha, Y, iter_times, weight_decay):
    num_samples = Kernel_Matrix.shape[0]
    IDs = np.random.rand(batch_size, 1) * num_samples
    IDs = IDs.astype(int).reshape(-1)
    haty_IDs = (Kernel_Matrix[IDs].dot(alpha * Y)) / (iter_times * weight_decay)
    mask = Y[IDs] * haty_IDs < 1
    alpha_copy = alpha.copy()
    alpha_copy[IDs] += mask
    # cause the dream brings back all the memories, the memories we have been through

    W_product_before = (alpha * Y).T.dot(Kernel_Matrix.dot(alpha * Y))
    W_product_after = (alpha_copy * Y).T.dot(Kernel_Matrix.dot(alpha_copy * Y))

    loss_before = 0.5 * weight_decay * np.sum(W_product_before.diagonal())
    loss_after = 0.5 * weight_decay * np.sum(W_product_after.diagonal())

    hatY_before = Kernel_Matrix.dot(alpha * Y) / (iter_times * weight_decay)
    hatY_after = Kernel_Matrix.dot(alpha_copy * Y) / (iter_times * weight_decay)

    mask_before = Y * hatY_before > 0
    mask_after = Y * hatY_after > 0

    l1_loss_before = np.sum(1 - (Y * hatY_before)[mask_before])
    l1_loss_after = np.sum(1 - (Y * hatY_after)[mask_after])

    loss_before += l1_loss_before
    loss_after += l1_loss_after

    if loss_after < loss_before:
        alpha = alpha_copy

    return alpha


def Split_Kernel_Pegasos(batch_size, Kernel_Matrix, alpha, Y, iter_times, weight_decay):
    # print(iter_times)

    ###################
    # Define parameters
    ###################
    num_samples = Kernel_Matrix.shape[0]
    num_tasks = Y.shape[1]

    #######################
    # Define the mini-batch
    #######################
    IDs = np.random.rand(batch_size, 1) * num_samples
    IDs = IDs.astype(int).reshape(-1)

    ###########################################
    # Make prediction using the old classifiers
    ###########################################
    haty_IDs = (Kernel_Matrix[IDs].dot(alpha * Y)) / (iter_times * weight_decay)

    ############################################################
    # Define two classifiers: before (update) and after (update)
    ############################################################
    mask = Y[IDs] * haty_IDs < 1
    alpha_copy = alpha.copy()
    alpha_copy[IDs] += mask

    W_product_before = (alpha * Y).T.dot(Kernel_Matrix.dot(alpha * Y))
    W_product_after = (alpha_copy * Y).T.dot(Kernel_Matrix.dot(alpha_copy * Y))
    loss_before = 0.5 * weight_decay * W_product_before.diagonal()
    loss_after = 0.5 * weight_decay * W_product_after.diagonal()

    hatY_before = Kernel_Matrix.dot(alpha * Y) / (iter_times * weight_decay)
    hatY_after = Kernel_Matrix.dot(alpha_copy * Y) / (iter_times * weight_decay)

    for i in range(num_tasks):
        mask_before = (Y[:, i] * hatY_before[:, i]) < 1
        mask_after = (Y[:, i] * hatY_after[:, i]) < 1

        l1_loss_before = np.sum(1 - (Y[:, i] * hatY_before[:, i])[mask_before])
        l1_loss_after = np.sum(1 - (Y[:, i] * hatY_after[:, i])[mask_after])

        loss_before[i] += l1_loss_before
        loss_after[i] += l1_loss_after

        if loss_after[i] < loss_before[i]:
            alpha[:, i] = alpha_copy[:, i]

    return alpha


def new_C2(Kernel_Matrix, Y, alpha, beta, batch_size, iter_times, weight_decay, checking=False):
    ###################
    # Define parameters
    ###################
    num_samples = Kernel_Matrix.shape[0]
    num_tasks = Y.shape[1]

    #######################
    # Define the mini-batch
    #######################
    IDs = np.random.rand(batch_size, 1) * num_samples
    IDs = IDs.astype(int).reshape(-1)

    ###########################################
    # Make prediction
    ###########################################

    haty_IDs = (Kernel_Matrix[IDs].dot(alpha * Y))
    for i in range(num_samples):
        hstack_M_K_i = Kernel_Matrix[i, IDs].repeat(num_tasks).reshape(batch_size, num_tasks)
        haty_IDs += hstack_M_K_i.dot(beta[i, :, :])
    haty_IDs /= (iter_times * weight_decay)

    # update alpha

    if checking:
        mask = Y[IDs] * haty_IDs < 1
        alpha_copy = alpha.copy()
        alpha_copy[IDs] += mask
        # cause the dream brings back all the memories, the memories we have been through

        W_product_before = (alpha * Y).T.dot(Kernel_Matrix.dot(alpha * Y))
        W_product_after = (alpha_copy * Y).T.dot(Kernel_Matrix.dot(alpha_copy * Y))

        loss_before = 0.5 * weight_decay * np.sum(W_product_before.diagonal())
        loss_after = 0.5 * weight_decay * np.sum(W_product_after.diagonal())

        hatY_before = Kernel_Matrix.dot(alpha * Y) / (iter_times * weight_decay)
        hatY_after = Kernel_Matrix.dot(alpha_copy * Y) / (iter_times * weight_decay)

        mask_before = Y * hatY_before > 0
        mask_after = Y * hatY_after > 0

        l1_loss_before = np.sum(1 - (Y * hatY_before)[mask_before])
        l1_loss_after = np.sum(1 - (Y * hatY_after)[mask_after])

        loss_before += l1_loss_before
        loss_after += l1_loss_after

        if loss_after < loss_before:
            alpha = alpha_copy

    if not checking:
        mask = Y[IDs] * haty_IDs < 1
        alpha[IDs] += mask

    ##update beta
    for i in range(len(IDs)):
        if np.argwhere(haty_IDs[i, :] < 0) != []:
            [neg_idx_list] = np.argwhere(haty_IDs[i, :] < 0)
            first_minus = neg_idx_list[0]
            [pos_idx_list] = np.argwhere(haty_IDs[i, first_minus:] > 0)

            pos_vec = np.zeros(num_tasks)
            neg_vec = np.zeros(num_tasks)
            pos_vec[pos_idx_list] = 1
            neg_vec[neg_idx_list] = 1

            matrix = np.outer(neg_vec, pos_vec)

            lower_tri = np.tril_indices(num_tasks)

            matrix[lower_tri] = 0

            for t in range(num_tasks):
                matrix[t, t] = (-1) * np.sum(matrix[t + 1:])

            beta[i] += matrix

        '''
        if np.argwhere(haty_IDs[i, :] < 0) != []:
            [negative_idx_list] = np.argwhere(haty_IDs[i, :] < 0)
            for negative_idx in negative_idx_list:
                [pos_idx_list] = np.argwhere(haty_IDs[i, negative_idx:] > 0)
                if pos_idx_list:
                    beta[ID[i], negative_idx, pos_idx_list] += 1
                beta[IDs[i], negative_idx, negative_idx] += (-1) * len(pos_idx_list)
        '''

    return alpha, beta
