CLASS z_humaneval_052 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: below_threshold
      IMPORTING
        l             TYPE int4_table
        t             TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_052 IMPLEMENTATION.

  METHOD below_threshold.
    LOOP AT l INTO DATA(e).
      IF e >= t.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDLOOP.
    result = abap_true.
  ENDMETHOD.

ENDCLASS.