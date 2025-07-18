CLASS z_humaneval_072 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: will_it_fly
      IMPORTING
        q             TYPE int4_table
        w             TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_072 IMPLEMENTATION.

  METHOD will_it_fly.
    DATA: sum   TYPE i,
          i     TYPE i,
          j     TYPE i,
          lines TYPE i.

    LOOP AT q INTO DATA(value).
      sum = sum + value.
    ENDLOOP.

    IF sum > w.
      result = abap_false.
      RETURN.
    ENDIF.

    lines = lines( q ).
    i = 1.
    j = lines.

    WHILE i < j.
      READ TABLE q INDEX i INTO DATA(left_value).
      READ TABLE q INDEX j INTO DATA(right_value).

      IF left_value <> right_value.
        result = abap_false.
        RETURN.
      ENDIF.

      i = i + 1.
      j = j - 1.
    ENDWHILE.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.