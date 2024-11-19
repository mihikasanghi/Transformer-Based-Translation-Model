import torch
from torch.utils.data import DataLoader
from utils import Transformer, train_tokenizer, TranslationDataset, collate_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def generate_translations(model, dataloader, tokenizer, device):
    model.eval()
    translations = []
    reference_texts = []
    
    with torch.no_grad():
        for source, target in tqdm(dataloader, desc="Generating translations"):
            source = source.to(device)
            output = model(source, source)
            output = torch.argmax(output, dim=-1)
            output = output.cpu().numpy()
            
            for idx in range(output.shape[0]):
                translation = tokenizer.decode(output[idx].tolist())
                end_token_idx = translation.find("[END]")
                if end_token_idx != -1:
                    translation = translation[:end_token_idx]
                translations.append(translation)
                
                reference = tokenizer.decode(target[idx].tolist())
                reference_texts.append(reference)
    
    return translations, reference_texts

def calculate_bleu_scores(reference_file, candidate_file):
    with open(reference_file, 'r', encoding='utf-8') as ref_file, \
         open(candidate_file, 'r', encoding='utf-8') as cand_file:
        reference_sentences = ref_file.readlines()
        candidate_sentences = cand_file.readlines()
    
    if len(reference_sentences) != len(candidate_sentences):
        raise ValueError("The number of sentences in both files must be equal")
    
    results = []
    for i, (reference, candidate) in tqdm(enumerate(zip(reference_sentences, candidate_sentences)), 
                                          total=len(reference_sentences), 
                                          desc="Calculating BLEU scores"):
        reference_tokens = [word_tokenize(reference.strip().lower())]
        candidate_tokens = word_tokenize(candidate.strip().lower())
        score = sentence_bleu(reference_tokens, candidate_tokens, 
                              smoothing_function=SmoothingFunction().method1)
        results.append((i+1, reference.strip(), candidate.strip(), score))
    
    return results

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_SEQUENCE_LENGTH = 25
    BATCH_SIZE = 128
    MODEL_PATH = 'transformer.pth'
    TEST_EN_PATH = "ted-talks-corpus/test.en"
    TEST_FR_PATH = "ted-talks-corpus/test.fr"
    TRAIN_EN_PATH = "ted-talks-corpus/train.en"
    TRAIN_FR_PATH = "ted-talks-corpus/train.fr"
    
    # Prepare tokenizer and model
    tokenizer = train_tokenizer(TRAIN_EN_PATH, TRAIN_FR_PATH)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=1024,
        num_heads=4,
        num_layers=3,
        d_ff=2048,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dropout=0.1,
        device=device
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Prepare test dataset and dataloader
    test_dataset = TranslationDataset(TEST_EN_PATH, TEST_FR_PATH, tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id, MAX_SEQUENCE_LENGTH)
    )
    
    # Generate translations
    generated_translations, reference_texts = generate_translations(model, test_dataloader, tokenizer, device)
    
    # Write results to files
    with open("reference_translations.txt", "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in reference_texts)
    
    with open("generated_translations.txt", "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in generated_translations)
    
    # Calculate BLEU scores
    scores = calculate_bleu_scores('reference_translations.txt', 'generated_translations.txt')
    avg_bleu_score = sum(score for _, _, _, score in scores) / len(scores)
    print(f"Average BLEU score on test set: {avg_bleu_score:.4f}")

if __name__ == '__main__':
    main()