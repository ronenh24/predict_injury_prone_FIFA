import load_clean_fifa_data
import pred_injury_prone_fifa


def pipeline():
    print("Load and Clean FIFA Data")
    load_clean_fifa_data.main()
    print()

    print("Predict Injury Prone")
    pred_injury_prone_fifa.main()


def main():
    pipeline()


if __name__ == "__main__":
    main()
