CLASS z_humaneval_127 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: intersection
      IMPORTING
        interval1     TYPE int4_table
        interval2     TYPE int4_table
      RETURNING
        VALUE(result) TYPE string.

  PRIVATE SECTION.
    CLASS-METHODS: is_prime
      IMPORTING
        num           TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_127 IMPLEMENTATION.

  METHOD intersection.
    DATA: l      TYPE i,
          r      TYPE i,
          length TYPE i.

    READ TABLE interval1 INDEX 1 INTO l.
    READ TABLE interval2 INDEX 1 INTO DATA(temp).
    IF temp > l.
      l = temp.
    ENDIF.

    READ TABLE interval1 INDEX 2 INTO r.
    READ TABLE interval2 INDEX 2 INTO temp.
    IF temp < r.
      r = temp.
    ENDIF.

    length = r - l.

    IF length > 0 AND is_prime( length ) = abap_true.
      result = 'YES'.
    ELSE.
      result = 'NO'.
    ENDIF.
  ENDMETHOD.

  METHOD is_prime.
    DATA: i TYPE i.

    IF num = 1 OR num = 0.
      result = abap_false.
      RETURN.
    ENDIF.

    IF num = 2.
      result = abap_true.
      RETURN.
    ENDIF.

    DO num - 2 TIMES.
      i = sy-index + 1.
      IF num MOD i = 0.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.