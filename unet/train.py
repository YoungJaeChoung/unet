# """TODO
# [ ] train 함수 리팩토링 ...
#     [ ] 다른데서는 보통 train 을 어떻게 구현하는지 검토
# """
# # 훈련 파라미터 설정하기
# lr = 1e-3
# batch_size = 4
# num_epoch = 20

# base_dir = './drive/MyDrive/DACrew/unet'
# data_dir = dir_data
# ckpt_dir = os.path.join(base_dir, "checkpoint")
# log_dir = os.path.join(base_dir, "log")


# # 훈련을 위한 Transform과 DataLoader
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
# loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

# dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
# loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# # 네트워크 생성하기
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = UNet().to(device)

# # 손실함수 정의하기
# fn_loss = nn.BCEWithLogitsLoss().to(device)

# # Optimizer 설정하기
# optim = torch.optim.Adam(net.parameters(), lr=lr)

# # 그밖에 부수적인 variables 설정하기
# num_data_train = len(dataset_train)
# num_data_val = len(dataset_val)

# num_batch_train = np.ceil(num_data_train / batch_size)
# num_batch_val = np.ceil(num_data_val / batch_size)

# # 그 밖에 부수적인 functions 설정하기
# fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
# fn_denorm = lambda x, mean, std: (x * std) + mean
# fn_class = lambda x: 1.0 * (x > 0.5)

# # Tensorboard 를 사용하기 위한 SummaryWriter 설정
# writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
# writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# # 네트워크 학습시키기
# st_epoch = 0
# # 학습한 모델이 있을 경우 모델 로드하기
# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# for epoch in range(st_epoch + 1, num_epoch + 1):
#         net.train()
#         loss_arr = []

#         for batch, data in enumerate(loader_train, 1):
#             # forward pass
#             label = data['label'].to(device)
#             input = data['input'].to(device)

#             output = net(input)

#             # backward pass
#             optim.zero_grad()

#             loss = fn_loss(output, label)
#             loss.backward()

#             optim.step()

#             # 손실함수 계산
#             loss_arr += [loss.item()]

#             print(
#                 "TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
#                 (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)),
#             )

#             # Tensorboard 저장하기
#             label = fn_tonumpy(label)
#             input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
#             output = fn_tonumpy(fn_class(output))

#             writer_train.add_image(
# 'label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
#             writer_train.add_image(
# 'input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
#             writer_train.add_image(
# 'output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

#         writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

#         with torch.no_grad():
#             net.eval()
#             loss_arr = []

#             for batch, data in enumerate(loader_val, 1):
#                 # forward pass
#                 label = data['label'].to(device)
#                 input = data['input'].to(device)

#                 output = net(input)

#                 # 손실함수 계산하기
#                 loss = fn_loss(output, label)

#                 loss_arr += [loss.item()]

#                 print(
#                     "VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
#                     (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)),
#                 )

#                 # Tensorboard 저장하기
#                 label = fn_tonumpy(label)
#                 input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
#                 output = fn_tonumpy(fn_class(output))

#                 writer_val.add_image(
# 'label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
#                 writer_val.add_image(
# 'input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
#                 writer_val.add_image(
# 'output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

#         writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

#         # epoch 50마다 모델 저장하기
#         if epoch % 50 == 0:
#             save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

#         writer_train.close()
#         writer_val.close()
