

# import time
# s = time.time()
# pbar = tqdm(train_loader, total=len(train_loader))
# ss = s
# for step, data in enumerate(pbar):
#     print(data['frames'].shape)
#     print(data['body'].shape)
#     print(data['ecg'].shape)
#     print(data['sub_lbls'].shape)
    
#     print(time.time()-ss)
#     ss = time.time()
#     print('\n')
    
# print(time.time()-s)

from sklearn.metrics import confusion_matrix, classification_report

# all_preds = np.asarray(all_preds).reshape(-1, 5)
# all_lbls = np.asarray(all_lbls).reshape(-1,)

# matrix = confusion_matrix(all_lbls, np.argmax(all_preds, axis=1), normalize='true',
#                           labels=[0,1,2,3,4])
# report = classification_report(all_lbls, np.argmax(all_preds, axis=1),
#                                 output_dict=True,
#                                 zero_division=0)

# # %%
# num_repeats_per_epoch = 4

# all_preds, all_lbls = [], []
# for i in range(10):
# # if (epoch + 1) % 10 == 0: # eval every 2 epoch
#     model.eval() # <-set mode important
#     test_acc, preds, lbls = [], [], []
#     # vbar = tqdm(test_loader, total= len(test_loader)*45) # total
#     for _ in range(num_repeats_per_epoch):
        
#         for step, test_batch in enumerate(test_loader):
#             sub_acc, sup_acc, preds, lbl_batch = evaluator.eval_step(test_batch)
#             test_acc.append(sub_acc)
#             # vbar.set_description(f'Validation - Acc {sub_acc:.4f}')
#             all_preds.append(preds)
#             all_lbls.append(lbl_batch)
#             print(test_batch['sub_lbls'])
#             # print(test_batch['filename'])
#     # print(f'Epoch: {epoch+1} => Average Acc: {np.nanmean(test_acc):.4f}')
# print(np.unique(all_lbls, return_counts=True))