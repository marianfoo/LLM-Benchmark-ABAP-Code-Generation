CLASS ltcl_triangle_area_test DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS test_case_1 FOR TESTING.
    METHODS test_case_2 FOR TESTING.
    METHODS test_case_3 FOR TESTING.
ENDCLASS.

CLASS ltcl_triangle_area_test IMPLEMENTATION.
  METHOD test_case_1.
    DATA(result) = z_humaneval_045=>triangle_area( a = 5 h = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '7.5' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA(result) = z_humaneval_045=>triangle_area( a = 2 h = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '2.0' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA(result) = z_humaneval_045=>triangle_area( a = 10 h = 8 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '40.0' ).
  ENDMETHOD.
ENDCLASS.