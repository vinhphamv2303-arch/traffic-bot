try:
    from data_preprocessing.legal_parser_modular.legal_parser.body.cli import main
except ModuleNotFoundError:
    from legal_parser.body.cli import main


if __name__ == "__main__":
    main()
