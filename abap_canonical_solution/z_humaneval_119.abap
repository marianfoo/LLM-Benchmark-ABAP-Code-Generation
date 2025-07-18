CLASS z_humaneval_119 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: match_parens
      IMPORTING
        VALUE(lst)    TYPE string_table
      RETURNING
        VALUE(result) TYPE string.

  PRIVATE SECTION.
    CLASS-METHODS: check
      IMPORTING
        VALUE(s)        TYPE string
      RETURNING
        VALUE(is_valid) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_119 IMPLEMENTATION.

  METHOD match_parens.
    DATA: s1 TYPE string,
          s2 TYPE string.

    IF lines( lst ) >= 2.
      READ TABLE lst INTO s1 INDEX 1.
      READ TABLE lst INTO s2 INDEX 2.
    ENDIF.

    DATA: combined1 TYPE string,
          combined2 TYPE string.

    CONCATENATE s1 s2 INTO combined1.
    CONCATENATE s2 s1 INTO combined2.

    IF check( combined1 ) = abap_true OR check( combined2 ) = abap_true.
      result = 'Yes'.
    ELSE.
      result = 'No'.
    ENDIF.
  ENDMETHOD.

  METHOD check.
    DATA: val  TYPE i,
          len  TYPE i,
          idx  TYPE i,
          char TYPE c.

    val = 0.
    len = strlen( s ).

    DO len TIMES.
      idx = sy-index - 1.
      char = s+idx(1).

      IF char = '('.
        val = val + 1.
      ELSE.
        val = val - 1.
      ENDIF.

      IF val < 0.
        is_valid = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    IF val = 0.
      is_valid = abap_true.
    ELSE.
      is_valid = abap_false.
    ENDIF.
  ENDMETHOD.

ENDCLASS.