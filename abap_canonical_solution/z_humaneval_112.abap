CLASS z_humaneval_112 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_result,
             result_string TYPE string,
             is_palindrome TYPE abap_bool,
           END OF ty_result.

    CLASS-METHODS: reverse_delete
      IMPORTING
        s             TYPE string
        c             TYPE string
      RETURNING
        VALUE(result) TYPE ty_result.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_112 IMPLEMENTATION.

  METHOD reverse_delete.
    DATA: lv_char     TYPE c LENGTH 1,
          lv_filtered TYPE string,
          lv_reversed TYPE string,
          lv_len      TYPE i,
          lv_i        TYPE i.

    DO strlen( s ) TIMES.
      lv_i = sy-index - 1.
      lv_char = s+lv_i(1).
      IF find( val = c sub = lv_char ) = -1.
        CONCATENATE lv_filtered lv_char INTO lv_filtered.
      ENDIF.
    ENDDO.

    lv_len = strlen( lv_filtered ).
    DO lv_len TIMES.
      lv_i = lv_len - sy-index.
      lv_char = lv_filtered+lv_i(1).
      CONCATENATE lv_reversed lv_char INTO lv_reversed.
    ENDDO.

    result-result_string = lv_filtered.
    IF lv_filtered = lv_reversed.
      result-is_palindrome = abap_true.
    ELSE.
      result-is_palindrome = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.