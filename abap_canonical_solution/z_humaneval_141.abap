CLASS z_humaneval_141 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: file_name_check
      IMPORTING
        file_name     TYPE string
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_141 IMPLEMENTATION.

  METHOD file_name_check.
    DATA: lv_dot_count      TYPE i,
          lv_digit_count    TYPE i,
          lv_pos            TYPE i,
          lv_char           TYPE c,
          lv_prefix         TYPE string,
          lv_suffix         TYPE string,
          lt_valid_suffixes TYPE string_table.

    lt_valid_suffixes = VALUE #( ( |txt| ) ( |exe| ) ( |dll| ) ).

    lv_dot_count = 0.
    DO strlen( file_name ) TIMES.
      lv_pos = sy-index - 1.
      lv_char = file_name+lv_pos(1).
      IF lv_char = '.'.
        lv_dot_count = lv_dot_count + 1.
      ENDIF.
    ENDDO.

    IF lv_dot_count <> 1.
      result = 'No'.
      RETURN.
    ENDIF.

    SPLIT file_name AT '.' INTO lv_prefix lv_suffix.

    READ TABLE lt_valid_suffixes TRANSPORTING NO FIELDS WITH KEY table_line = lv_suffix.
    IF sy-subrc <> 0.
      result = 'No'.
      RETURN.
    ENDIF.

    IF strlen( lv_prefix ) = 0.
      result = 'No'.
      RETURN.
    ENDIF.

    lv_char = lv_prefix+0(1).
    IF NOT ( ( lv_char >= 'a' AND lv_char <= 'z' ) OR ( lv_char >= 'A' AND lv_char <= 'Z' ) ).
      result = 'No'.
      RETURN.
    ENDIF.

    lv_digit_count = 0.
    DO strlen( lv_prefix ) TIMES.
      lv_pos = sy-index - 1.
      lv_char = lv_prefix+lv_pos(1).
      IF lv_char >= '0' AND lv_char <= '9'.
        lv_digit_count = lv_digit_count + 1.
      ENDIF.
    ENDDO.

    IF lv_digit_count > 3.
      result = 'No'.
      RETURN.
    ENDIF.

    result = 'Yes'.

  ENDMETHOD.

ENDCLASS.