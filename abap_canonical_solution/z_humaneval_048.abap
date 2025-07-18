CLASS z_humaneval_048 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: is_palindrome
      IMPORTING
        text          TYPE string
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_048 IMPLEMENTATION.

  METHOD is_palindrome.
    DATA: len   TYPE i,
          i     TYPE i,
          char1 TYPE c LENGTH 1,
          char2 TYPE c LENGTH 1.

    len = strlen( text ).

    DO len TIMES.
      i = sy-index - 1.
      char1 = substring( val = text off = i len = 1 ).
      char2 = substring( val = text off = ( len - 1 - i ) len = 1 ).
      IF char1 <> char2.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.