try:
    from data_preprocessing.legal_parser_modular.legal_parser.package.cli import main
except ModuleNotFoundError:
    from legal_parser.package.cli import main


if __name__ == "__main__":
    main()
