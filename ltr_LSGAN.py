



def trainingStep(i,  data, 
                 real_label, fake_label, 
                 netD, netG, 
                 device, nz, optimizerD, optimizerG, criterion):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    # Format batch
    
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)

    label = torch.full((b_size,), real_label, device=device)
    # Forward pass real batch through D
    D_real = netD(real_cpu).view(-1)

    ## Train with all-fake batch
    # Generate batch of latent vectors
    z = torch.randn(b_size, nz, 1, 1, device=device)

   # Generate fake image batch with G
    G_sample = netG(z)
    # Classify all fake batch with D
    D_fake = netD(G_sample.detach()).view(-1)

    label.fill_(fake_label)

    D_loss = 0.5 * (torch.mean((D_real - 1)**2) + torch.mean(D_fake**2) )

    # Calculate the gradients for this batch
    D_loss.backward()
    optimizerD.step()


    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    D_fake = netD(G_sample).view(-1)

    G_loss = 0.5 * torch.mean((D_fake -1)**2 )
    
    G_loss.backward()
    optimizerG.step()
    
    return D_loss, G_loss


#---------------------------------------------------------
def stringaStato(epoch, num_epochs, i, dataloader, D_loss, G_loss):
    out = ('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch, num_epochs, 
                        i, len(dataloader),
                        # Loss_D    Loss_G
                        D_loss.item(), G_loss.item()
                        )
          )
    return out


#-------------------------------------------------
def salvaCSV(lista, nomefile):
# salva le liste come file csv
    with open(nomefile, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(lista)
