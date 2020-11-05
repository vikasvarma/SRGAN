import csv
from trainer import SRGANTrainer

# Setup:
trainer = SRGANTrainer(
    train_data = './train_data/',
    test_data  = './test_data/'
)

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
                
        finally:
            f.close()
