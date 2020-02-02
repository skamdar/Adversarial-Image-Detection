# targeted attack
def fgsm_attack(model, data, target, epsilon):
   
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    #print(init_pred)
    # If the initial prediction is wrong, dont bother attacking, just move on
    if init_pred.item() != target.item():
        return data

    # Calculate the loss
    loss = F.nll_loss(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = data - epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # Return the perturbed image
    return perturbed_image


def i_fgsm_attack(model, data, target, epsilon, itr):
   
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    #print(init_pred)
    # If the initial prediction is wrong, dont bother attacking, just move on
    if init_pred.item() != target.item():
        return data
    for i in range(itr):
        data = Variable(data.data, requires_grad=True)
        output = model(data)
        # Calculate the loss
        loss = F.nll_loss(output, target)
        
        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        data = data - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        data = torch.clamp(data, 0, 1)
        
    # Return the perturbed image
    return data



def gn_adv(args, model, device, target):
    """

    :param args:
    :param model:
    :param device:
    :param target: label for which adv image is to be generated
    :return: adv image
    """
    model.train()
    target = torch.Tensor([target]).long()
    #data = torch.rand((1, 1, 28, 28), requires_grad=True, device=device) # uncomment this line for cnn
    data = torch.rand((1, 784), requires_grad=True, device=device) # comment this line while using cnn
    data, target = data.to(device), target.to(device)
    optimizer = optim.SGD([data], lr=1, momentum=args.momentum)

    for itr in range(50):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        #print(loss)
        optimizer.step()

    return torch.squeeze(data).detach().cpu().numpy()


def sneaky_adv(args, model, device, img, target, lmbd):
    """
    :param img: image whose alike image to be constructed
    :param target: label that network should output
    :return: image that looks like img and network output is target
    """
    model.train()
    #data = torch.rand((1, 1, 28, 28), device=device, requires_grad=True)
    data = torch.rand((1, 784), requires_grad=True, device=device) # comment this line while using cnn
    target = torch.Tensor([target]).long()
    #img = torch.Tensor(img).view((1, 1, 28, 28))
    img = torch.Tensor(img).view((1, 784))
    img, target, data = img.to(device), target.to(device), data.to(device)
    optimizer = optim.SGD([data], lr=0.5, momentum=args.momentum)

    for itr in range(50):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) + lmbd*torch.norm((img-data), 2)
        loss.backward()
        #print(loss)
        optimizer.step()

    return torch.squeeze(data).detach().cpu().numpy()
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = data.reshape(-1, 784) # Remove this line while training cnn
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data = data.reshape(-1, 784) # Remove this line while training cnn
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def kld(model, actual):
    """
    model : its the array that is input to the network
    actual : image corresponding to the network output
    return: KL divergence between two prob distributions
    """
    model = np.asarray(list(model) + [0 for i in range(255 - len(model))])
    actual = np.asarray(list(actual) + [0 for i in range(255 - len(actual))])
    k = ((model * np.log(model)) - (model * np.log(actual))).sum()
    print(k)
    # k = (model * ma.log(model/actual)).sum()

    return k


def mean_image(tr_data):
    """

    :param tr_data: input mnist data
    :return: mean images for each class
    """
    val = {}
    for i in range(10):
        mat = []
        for r in range(len(tr_data)):
            if tr_data[r][1][i] == 1:
                x = tr_data[r][0]
                mat.append(np.asarray(x))
        val[i] = np.mean(mat, axis=0)

    return val


def gn_idx_list(training_data):
    idx_list = {}
    for i in range(10):
        idx_list[i] = []
    for i in range(len(training_data)):
        idx_list[np.argmax(training_data[i][1])].append(i)

    return idx_list    



def gn_adv_imgs(args, model, device, training_data):
    """

    :param training_data:
    :return: 100 adversarial images
    """
    gn = {}
    """idx_list = []
    for i in range(10):
        idx = np.random.randint(0, 8000)
        while training_data[idx][1][i] != 1:
            idx += 1
        idx_list.append(idx)"""
    idx_list = gn_idx_list(training_data)

    for g in range(10):
        gl = []
        print(g)
        for v in range(10):
            if g != v:
                for i in range(4):
                    idx = idx_list[v][i]
                    gl.append(sneaky_adv(args, model, device, training_data[idx][0], g, 0.4))
            else:
                for i in range(4):
                    idx = idx_list[v][i]
                    gl.append(gn_adv(args, model, device, g))
        gn[g] = gl

    return gn


def gn_pd_imgs(adv_imgs):
    """
    adv_imgs: a dictionary with keys as the classes and corresponding adv images as values
    returns gn_pd : a dictionary with keys as classes and corresponding prob. distribution of adv images
    """
    gn_pd = {}
    for k, v in adv_imgs.items():
        bin_y = []

        for i in range(len(v)):
            bin_y.append(generate_pd(v[i]))

        gn_pd[k] = bin_y

    return gn_pd


def mean_pd(mean_imgs):
    """
    val: a dictionary with keys as the classes and corresponding mean images as values
    returns val_pd : a dictionary with keys as classes and corresponding prob. distribution of mean images
    """
    val_pd = {}
    for k, v in mean_imgs.items():
        # print(k)
        # print(len(v))
        val_pd[k] = generate_pd(v)

    return val_pd


def generate_pd(img, patch_size=28):
    """
    img: an image
    returns pd : prob distribution of intensities in img
    """
    
    if img.max() - img.min() == 0:
        img = np.zeros((patch_size, patch_size))
    else:
        img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255).astype('uint8'))
    img = np.floor(img)
    img = img.reshape((-1, patch_size**2))
    img = img.astype('int64')
    bin_m = np.bincount(img[0])
    l1 = list(bin_m)
    l2 = [0]*(256 - len(bin_m))
    bin_m = l1 + l2
    bin_m = [x if x != 0 else 0.0001 for x in bin_m] 
    bin_m = bin_m / sum(bin_m)
    # making sure to have len of list 255
    pd = np.asarray(bin_m)
    #pd = [x if x != 0 else 0.0001 for x in pd]  # to avoid divide by zero in KLD

    return pd



def kl_calc(adv_im, mean_im, win_size):
    """
    """
    adv_im = np.asarray(adv_im).reshape(28, 28)
    mean_im = np.asarray(mean_im).reshape(28, 28)
    k_sum = 0
    win_num = int(28 / win_size)
    
    for i in range(win_num):
        for j in range(win_num):
            s_arr = adv_im[i: i + win_size, i: i + win_size]
            p = generate_pd(s_arr, win_size)
            # mean image pd
            ms_arr = mean_im[i: i + win_size, i: i + win_size]
            q = generate_pd(ms_arr, win_size)
            k_sum += scipy.stats.entropy(p, q)

    return k_sum

def js_calc(adv_im, mean_im, win_size):
    """
    """
    adv_im = np.asarray(adv_im).reshape(28, 28)
    mean_im = np.asarray(mean_im).reshape(28, 28)
    k_sum = 0
    num_win = int(28 / win_size)
    
    for i in range(num_win):
        for j in range(num_win):
            s_arr = adv_im[i: i + win_size, j: j + win_size]
            p = generate_pd(s_arr, win_size)
            # mean image pd
            ms_arr = mean_im[i: i + win_size, j: j + win_size]
            q = generate_pd(ms_arr, win_size)
            r = (p + q)/ 2
            k_sum += (scipy.stats.entropy(p, r) / 2) + (scipy.stats.entropy(q, r) / 2)

    return k_sum

def ac_calc(training_data, adv_imgs, mean_imgs, idx_lst, fun, win_size):
    """
    
    """
    tr_l = []
    ad_l = []
    for j in range(10):
        lst = idx_l[j]
        for i in range(4000):
            ind = lst[i]
            tr_l.append(fun(training_data[ind][0], mean_imgs[j], win_size))
        for i in range(4000):
            ad_l.append(fun(adv_imgs[j][i], mean_imgs[j], win_size))

    y = [0]*len(ad_l) + [1]*len(tr_l)
    x = ad_l + tr_l
    x2 = [i**2 for i in x]
    x3 = [i**3 for i in x]
    data = {'x': x, 'x2': x2, 'x3': x3, 'y': y}
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)                   
    n_df = df[['x', 'x2', 'x3']]
    normalized_df = (n_df - n_df.mean()) / n_df.std()
    normalized_df['y'] = df['y']


    logreg = LogisticRegression()

    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg.fit(normalized_df[['x', 'x2', 'x3']][:60000], normalized_df['y'][:60000])

    res = logreg.predict(normalized_df[['x', 'x2', 'x3']][60000:])
    Y = np.asarray(normalized_df['y'][60000:])
    acc = (np.sum(Y == res) / 20000)*100
    print(acc)                   


def min_thr_calc(adv_imgs, mean_imgs, fun):
    """
    # each class contain 4000 adversarial examples
    # 400 uniformly chosen samples out of 4000 are used for min_thr calc
    return: list containing min_thr for each of the 10 classes
    """
    min_thr = []
    ind = np.random.randint(40, size=4)
    tmp = 1000
    for i in range(len(adv_imgs)):
        f = adv_imgs[i]
        for j in ind:
            tmp = min(tmp, fun(f[j], mean_imgs[i]))
        if (tmp == 1000):
            print(i)
        min_thr.append(tmp) 
        tmp =1000
    
    return min_thr

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Target Label (%d)' % i for i in range(0,n_images)]
    fig = plt.figure(frameon=False)
    for n, (image, title) in enumerate(zip(images, titles)):
        
        #a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a = fig.add_subplot(cols, np.ceil(n_images/float(1)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.axis('off')    
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    
# to combine images vertically
def combine_imgs(list_im):
    #list_im = ['type1_advf_imgs.png', 'type2_advf_imgs.png']
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save( 'Trifecta_vertical.png' )

