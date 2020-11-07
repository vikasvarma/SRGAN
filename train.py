import csv
from trainer import SRGANTrainer

# Setup:
trainer = SRGANTrainer(data_folder='./train-data/')

if __name__ == '__main__':

    with open('train_results.csv', 'wt') as f:
        try:
            writer = csv.writer(f)
            writer.writerow( ('EPOCH', 'G_LOSS', 'D_LOSS', 'SSIM', 'PSNR') )
            # Run each epoch:
            for epoch in range(trainer.num_epochs):
                # Run epoch and gather mean results:
                gen_loss, disc_loss, ssim, psnr = trainer.doepoch(epoch)
                
                # Save results to a CSV:
                writer.writerow( (epoch + 1, gen_loss, disc_loss, ssim, psnr) )
                
                # Save state:
                trainer.save(epoch)

        finally:
            f.close()
            
    # Now, evaluate the model against the test dataset:
    ssim, psnr = trainer.evaluate(data_folder='./test-data/')
    print('Completed Training. SSIM: %.4f, PSNR %.4f db' % ssim, psnr)
