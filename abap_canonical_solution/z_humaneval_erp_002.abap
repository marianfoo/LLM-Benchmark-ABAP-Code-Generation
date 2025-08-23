CLASS z_humaneval_erp_002 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.

    CLASS-METHODS: count_subordinates
      IMPORTING
        supervisor_id TYPE char20
      RETURNING
        VALUE(rv_amount) TYPE i.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_erp_002 IMPLEMENTATION.

  METHOD count_subordinates.
    DATA: lt_org_data TYPE TABLE OF zorg_structure,
          lv_count    TYPE i.

    " Get organizational data from database table
    SELECT employee_id, supervisor_id
      FROM zorg_structure
      INTO TABLE @lt_org_data.

    " Count direct subordinates
    LOOP AT lt_org_data INTO DATA(ls_org_entry) WHERE supervisor_id = supervisor_id.
      lv_count = lv_count + 1.
    ENDLOOP.

    rv_amount = lv_count.

  ENDMETHOD.

ENDCLASS.