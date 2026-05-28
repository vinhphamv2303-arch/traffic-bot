try:
    from data_preprocessing.legal_parser_modular.legal_parser.attachments.cli import main
except ModuleNotFoundError:
    from legal_parser.attachments.cli import main


if __name__ == "__main__":
    main()
