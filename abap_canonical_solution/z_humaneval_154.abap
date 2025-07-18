CLASS z_humaneval_154 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: cycpattern_check
      IMPORTING
        VALUE(a)      TYPE string
        VALUE(b)      TYPE string
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_154 IMPLEMENTATION.

  METHOD cycpattern_check.
    DATA: l             TYPE i,
          pat           TYPE string,
          i             TYPE i,
          j             TYPE i,
          substring_a   TYPE string,
          substring_pat TYPE string.

    l = strlen( b ).
    pat = |{ b }{ b }|.

    DO strlen( a ) - l + 1 TIMES.
      i = sy-index - 1.
      substring_a = a+i(l).

      DO l + 1 TIMES.
        j = sy-index - 1.
        substring_pat = pat+j(l).

        IF substring_a = substring_pat.
          result = abap_true.
          RETURN.
        ENDIF.
      ENDDO.
    ENDDO.

    result = abap_false.
  ENDMETHOD.

ENDCLASS.